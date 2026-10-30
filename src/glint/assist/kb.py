"""Knowledge-base search results within a context budget, for the tools arm's KB_search, grep and KB_read
(glint.assist.tau2_tools.BankingDecisionTools).

τ²'s bm25_grep tools return whole documents: KB_search its 10 best, grep every match ranked by match count. In
banking-plan-v5-gated task_081 trial 0 made 16 KB_search/grep calls that returned 327,401 characters (one 29,860),
and trials 1-3 ended on Glimmer's 131,072-token context limit. Here each result keeps the 5 best documents, each as
its title, id and the one paragraph that best matches (at most 600 characters), at most 4,000 characters in all;
KB_read pages through a whole document 6,000 characters at a time. A context budget, not a policy rule.
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
    """The best TOP_DOCS documents of `results` (doc id, score; best first), each as title, id and best paragraph,
    within RESULT_CHARS; documents that do not fit are counted, not shown."""
    if not results:
        return empty
    out, shown = [], 0
    for i, (doc_id, _) in enumerate(results[:TOP_DOCS], 1):
        entry = (f"{i}. {pipeline.get_document_title(doc_id) or 'Untitled'}\n   ID: {doc_id}\n"
                 f"   Excerpt: {best_paragraph(pipeline.get_document_content(doc_id) or '', score)}\n")
        if len("\n".join(out + [entry])) + len(READ_HINT) + 80 > RESULT_CHARS:
            break
        out.append(entry)
        shown += 1
    more = len(results) - shown
    tail = f"{more} more matching document(s) not shown; narrow the query to see them. " if more > 0 else ""
    return "\n".join(out) + f"\n{tail}{READ_HINT}"


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
