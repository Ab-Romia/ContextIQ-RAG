"""Structure-aware chunking.

A chunk is only useful if it is small enough to be specific and large enough to stand
on its own. We split on document structure first (Markdown-style headings), then split
each section into token-bounded pieces with overlap. Carrying the heading path on every
chunk lets the next stage prepend real context, and lets a citation point at a section
rather than an anonymous offset.
"""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_into_sections(text: str) -> list[tuple[list[str], str, int]]:
    """Group lines into sections under their heading path.

    Returns (heading_path, section_text, start_offset). Text with no Markdown headings
    yields a single section with an empty path, which is the common case for PDFs and
    plain text and is handled the same way as a richly structured document.
    """
    sections: list[tuple[list[str], str, int]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    buffer: list[str] = []
    buffer_start = 0
    offset = 0

    def flush(start: int) -> None:
        body = "".join(buffer).strip()
        if body:
            path = [title for _, title in heading_stack]
            sections.append((path, body, start))
        buffer.clear()

    for line in text.splitlines(keepends=True):
        match = _HEADING.match(line.rstrip("\n"))
        if match:
            flush(buffer_start)
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            buffer_start = offset + len(line)
        else:
            if not buffer:
                buffer_start = offset
            buffer.append(line)
        offset += len(line)

    flush(buffer_start)
    return sections


def chunk_document(
    text: str,
    source: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
    )
    chunks: list[Chunk] = []
    ordinal = 0

    for heading_path, body, base_offset in _split_into_sections(text):
        cursor = 0
        for piece in splitter.split_text(body):
            stripped = piece.strip()
            if not stripped:
                continue
            # Locate the piece in the section so the citation can point back at the
            # exact span. Searching from the previous cursor keeps overlapping pieces
            # in document order even when their text repeats.
            local = body.find(piece, cursor)
            if local == -1:
                local = cursor
            start = base_offset + local
            cursor = local + 1
            chunks.append(
                Chunk(
                    id=f"{source}::{ordinal}",
                    text=stripped,
                    source=source,
                    ordinal=ordinal,
                    heading_path=heading_path,
                    char_span=(start, start + len(piece)),
                )
            )
            ordinal += 1

    return chunks
