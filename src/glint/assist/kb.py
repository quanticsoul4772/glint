"""Knowledge-base search results within a context budget, for the tools arm's KB_search, grep and KB_read
(glint.assist.tau2_tools.BankingDecisionTools).

τ²'s bm25_grep tools return whole documents: KB_search its 10 best, grep every match ranked by match count. In
banking-plan-v5-gated task_081 trial 0 made 16 KB_search/grep calls that returned 327,401 characters (one 29,860),
and trials 1-3 ended on Glimmer's 131,072-token context limit. Here each result keeps the 5 best documents: the best
one whole up to 6,000 characters (since 2026-10-31; see format_results), the others as title, id and the one
paragraph that best matches (at most 600 characters), at most 4,000 characters for those; KB_read pages through a
whole document 6,000 characters at a time. A context budget, not a policy rule.
Deterministic: same documents, same query, same text.
"""
from __future__ import annotations

import re

TOP_DOCS, EXCERPT_CHARS, RESULT_CHARS, PAGE_CHARS = 5, 600, 4000, 6000
READ_HINT = "Read a whole document with KB_read(doc_id)."
WORD = re.compile(r"[a-z0-9_]{3,}")


def paragraphs(content: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]


def best_paragraph(content: str, score) -> str:
    """The paragraph with the highest score (the first of equals), cut to EXCERPT_CHARS."""
    paras = paragraphs(content) or [content.strip()]
    best = max(paras, key=score)  # max keeps the first of equal scores
    return best if len(best) <= EXCERPT_CHARS else best[:EXCERPT_CHARS - 1].rstrip() + "…"


def query_score(query: str):
    terms = set(WORD.findall(query.lower()))
    return lambda para: len(terms & set(WORD.findall(para.lower())))


def pattern_score(pattern: str):
    try:
        rx = re.compile(pattern, re.IGNORECASE)  # τ²'s GrepRetriever: case-insensitive, a bad regex taken literally
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    return lambda para: len(rx.findall(para))


def format_results(pipeline, results: list[tuple[str, float]], score, empty: str) -> str:
    """The best TOP_DOCS documents of `results` (doc id, score; best first): the best one whole, up to PAGE_CHARS
    (KB_read pages the rest), and the others as title, id and best paragraph within RESULT_CHARS; documents that do
    not fit are counted, not shown.

    The best document is whole because excerpts alone left the agent searching again: in banking-unseen-gated the 41
    tools-arm runs that ran out of τ²'s 200 steps made 93 KB_search calls a run, nearly all reading only one
    paragraph per document, where Glimmer alone (10 whole documents a search) searched 18 times (2026-10-31)."""
    if not results:
        return empty
    out, shown = [], 0
    for i, (doc_id, _) in enumerate(results[:TOP_DOCS], 1):
        title, content = pipeline.get_document_title(doc_id) or "Untitled", pipeline.get_document_content(doc_id) or ""
        if i == 1:
            more = (f"\n   [continues: KB_read(doc_id={doc_id!r}, offset={PAGE_CHARS})]" if len(content) > PAGE_CHARS
                    else "")
            out.append(f"1. {title}\n   ID: {doc_id}\n   Document:\n{content[:PAGE_CHARS]}{more}\n")
            shown += 1
            continue
        entry = f"{i}. {title}\n   ID: {doc_id}\n   Excerpt: {best_paragraph(content, score)}\n"
        if len("\n".join(out[1:] + [entry])) + len(READ_HINT) + 80 > RESULT_CHARS:
            break
        out.append(entry)
        shown += 1
    more = len(results) - shown
    tail = f"{more} more matching document(s) not shown; narrow the query to see them. " if more > 0 else ""
    return "\n".join(out) + f"\n{tail}{READ_HINT}"


def diagnosis(pipeline, results: list[tuple[str, float]], returned: list[str], searches: int) -> str:
    """A compact search answer for a conversation past its search budget (appendix #72, reader 3 R6: "explicit
    diagnosis replaces accumulated search output"): titles and IDs only, documents an earlier search already returned
    named apart, and one line saying what the searching has found so far. A context budget, not a policy rule;
    used only when glint.assist.tau2_tools.SEARCH_COMPACT_AFTER is set (default 20 since the owner's decision of
    2026-11-03, every fix on by default; overflow occurred only in the alone arm, so it changes the tools arm's
    context handling, not the baseline's)."""
    top = [doc_id for doc_id, _ in results[:TOP_DOCS]]
    new = [d for d in top if d not in returned]
    lines = [f"Search {searches} in this conversation; {len(set(returned))} distinct documents returned so far."]
    if not top:
        lines.append("This search found no documents.")
    for doc_id in new:
        lines.append(f"- {pipeline.get_document_title(doc_id) or 'Untitled'} (ID: {doc_id})")
    if len(new) < len(top):
        lines.append(f"Already returned by an earlier search: {', '.join(d for d in top if d in returned)}.")
    lines.append("Diagnosis: more searching is unlikely to add policy. Read a listed document with KB_read(doc_id), "
                 "or act on the policy already found and tell the customer what is left.")
    return "\n".join(lines)


def read_page(pipeline, doc_id: str, offset: int = 0) -> str:
    """Characters offset to offset+PAGE_CHARS of a document, with where the next page starts."""
    content = pipeline.get_document_content(doc_id)
    if content is None:
        return f"No document with id {doc_id!r}; KB_search and grep results give each document's ID."
    offset = max(0, int(offset))
    page = content[offset:offset + PAGE_CHARS]
    end = offset + len(page)
    head = (f"{pipeline.get_document_title(doc_id) or 'Untitled'} (ID: {doc_id}), characters {offset}-{end} of "
            f"{len(content)}\n\n")
    nxt = f"\n\n[continues: KB_read(doc_id={doc_id!r}, offset={end})]" if end < len(content) else "\n\n[end of document]"
    return head + page + nxt
