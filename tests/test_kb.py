"""The tools arm's KB_search, grep and KB_read within a context budget (glint.assist.kb), on τ²'s banking knowledge
base; the base arm and glint-bank without decision tools keep τ²'s whole-document results."""
import re

import pytest
from loguru import logger
from tau2.registry import registry

from glint.assist import kb
from glint.assist.tau2_tools import BankingDecisionTools, assist_environment

logger.remove()
# searches a task_081-like conversation makes (stolen cards, fraud disputes, replacements)
SEARCHES = [("dispute unauthorized fraudulent charge", "fraud"), ("stolen card replacement", "stolen|replacement"),
            ("provisional credit criteria", "provisional credit"), ("card last 4 digits", "card_last_4_digits"),
            ("credit limit increase", "credit limit")]


def _env(assisted: bool):
    env = registry.get_env_constructor("banking_knowledge")(retrieval_variant="bm25_grep")
    return assist_environment(env, "banking_knowledge") if assisted else env


@pytest.mark.parametrize(("query", "pattern"), SEARCHES)
def test_search_results_stay_within_the_budget_and_name_their_documents(query, pattern):
    tools, base = _env(True).tools, _env(False).tools
    for name, args in (("KB_search", {"query": query}), ("grep", {"pattern": pattern})):
        out, full = tools.use_tool(name, **args), base.use_tool(name, **args)
        # the best document whole (up to PAGE_CHARS) plus the others' excerpts within RESULT_CHARS (2026-10-31)
        assert len(out) <= kb.PAGE_CHARS + kb.RESULT_CHARS + 400 and kb.READ_HINT in out
        ids = re.findall(r"   ID: (\S+)", out)
        assert 1 <= len(ids) <= kb.TOP_DOCS and ids == re.findall(r"   ID: (\S+)", full)[:len(ids)]  # τ²'s ranking
        assert all(len(e) <= kb.EXCERPT_CHARS for e in re.findall(r"   Excerpt: (.*)\n", out))
        assert len(full) > len(out)


def test_grep_excerpt_is_the_paragraph_with_the_most_matches():
    tools = _env(True).tools
    out = tools.use_tool("grep", pattern="card_last_4_digits")
    doc = re.findall(r"   ID: (\S+)", out)[1]  # the first document is shown whole; the second has the first excerpt
    content = tools._grep_pipeline.get_document_content(doc)
    excerpt = re.search(r"   Excerpt: (.*)\n", out)[1]
    best = max(kb.paragraphs(content), key=kb.pattern_score("card_last_4_digits"))
    assert "card_last_4_digits" in excerpt.lower() and best.startswith(excerpt.rstrip("…").strip()[:200])


def test_kb_read_pages_cover_the_whole_document():
    tools = _env(True).tools
    doc = re.search(r"   ID: (\S+)", tools.use_tool("KB_search", query="dispute unauthorized fraudulent charge"))[1]
    content = tools._kb_pipeline.get_document_content(doc)
    text, offset = "", 0
    while True:
        page = tools.use_tool("KB_read", doc_id=doc, offset=offset)
        body = page.split("\n\n", 1)[1].rsplit("\n\n[", 1)[0]
        assert len(body) <= kb.PAGE_CHARS
        text += body
        nxt = re.search(r"offset=(\d+)\)\]$", page)
        if nxt is None:
            assert page.endswith("[end of document]")
            break
        offset = int(nxt[1])
    assert text == content
    assert tools.use_tool("KB_read", doc_id="no_such_doc").startswith("No document with id")


def test_only_the_tools_arm_gets_the_budgeted_search():
    assisted, base = _env(True).tools, _env(False).tools
    assert isinstance(assisted, BankingDecisionTools) and not isinstance(base, BankingDecisionTools)
    assert "KB_read" in assisted.get_tools() and "KB_read" not in base.get_tools()
    assert "Content: " in base.use_tool("KB_search", query="stolen card replacement")  # τ²'s whole documents


def test_the_best_search_result_is_whole_and_a_repeat_points_back():
    # banking-unseen-gated, 2026-10-31: the tools-arm runs that ran out of τ²'s 200 steps made 93 KB_search calls a
    # run on one-paragraph excerpts, one in five repeating an earlier search exactly
    tools = _env(True).tools
    first = tools.use_tool("KB_search", query="stolen card replacement")
    top = first.split("   ID: ", 1)[1].split("\n", 1)[0]
    content = tools._kb_pipeline.get_document_content(top)
    assert content[:kb.PAGE_CHARS] in first  # the best document's text, not a paragraph of it
    if len(content) > kb.PAGE_CHARS:
        assert f"KB_read(doc_id={top!r}, offset={kb.PAGE_CHARS})" in first
    again = tools.use_tool("KB_search", query="  Stolen   CARD replacement ")
    assert again.startswith("Same KB_search as your search 1 in this conversation") and top in again
    assert len(again) < 600
    other = tools.use_tool("grep", pattern="stolen card replacement")  # another tool: not a repeat
    # numbered by call, repeats included: KB_search 1, its repeat 2, this grep 3
    assert not other.startswith("Same") and tools.use_tool("grep", pattern="stolen card replacement").startswith(
        "Same grep as your search 3")
    # a regex's case and spacing matter (review: \D and \d were one search); a repeat that found nothing says so
    assert not tools.use_tool("grep", pattern=r"\D{3}").startswith("Same")
    assert not tools.use_tool("grep", pattern=r"\d{3}").startswith("Same")
    nothing = tools.use_tool("grep", pattern="zzqxnomatchzz")
    assert tools.use_tool("grep", pattern="zzqxnomatchzz").endswith("it found no documents. Search with different "
                                                                   "words.") and "No matches" in nothing
    fresh = _env(True).tools  # a new conversation keeps no record
    assert not fresh.use_tool("KB_search", query="stolen card replacement").startswith("Same")


def test_the_run_log_warns_once_when_a_conversation_reaches_the_search_limit(capsys, monkeypatch):
    # banking-unseen-gated, 2026-10-31: every run that ran out of τ²'s 200 steps had made at least 72 searches, and
    # nothing in the run log said so until it ended
    from glint.assist import tau2_tools

    monkeypatch.setattr(tau2_tools, "SEARCH_WARN", 3)
    tools = _env(True).tools
    tools.use_tool("KB_search", query="stolen card replacement")
    tools.use_tool("grep", pattern="fraud")
    assert "WARNING" not in capsys.readouterr().err
    tools.use_tool("KB_read", doc_id="doc_credit_cards_credit_cards_(general)_018")
    tools.use_tool("KB_search", query="stolen card replacement")  # a repeat still counts as a call
    err = capsys.readouterr().err
    assert err.count("WARNING: a conversation has made 3 knowledge-base searches") == 1
