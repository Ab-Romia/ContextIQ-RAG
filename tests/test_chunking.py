from app.chunking import chunk_document

DOC = """# Handbook

Intro paragraph that sits before any subsection.

## Expenses

The meal cap is 55 credits per day. Approval is needed above 2500 credits.

## Security

All laptops use full disk encryption.
"""


def _chunks():
    return chunk_document(DOC, "handbook.md", chunk_tokens=64, overlap_tokens=8)


def test_chunks_carry_heading_path():
    chunks = _chunks()
    assert chunks, "expected at least one chunk"
    expenses = [c for c in chunks if "meal cap" in c.text]
    assert expenses, "expected a chunk containing the expenses text"
    assert expenses[0].heading_path == ["Handbook", "Expenses"]


def test_char_spans_point_back_into_the_source():
    for chunk in _chunks():
        start, end = chunk.char_span
        assert 0 <= start < end <= len(DOC)
        # the recorded span should overlap the chunk's own text
        assert chunk.text.split("\n")[0][:20] in DOC[start:end] or chunk.text[:20] in DOC[start:end]


def test_ordinals_are_unique_and_sequential():
    ordinals = [c.ordinal for c in _chunks()]
    assert ordinals == list(range(len(ordinals)))
