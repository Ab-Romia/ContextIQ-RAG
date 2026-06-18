"""Contextual augmentation.

A retrieved chunk often reads like an orphan: "It raised the limit to 40" means
nothing without knowing the document and section it came from. Before embedding, we
prepend a short header built from the source title and heading path so the vector
captures that context. The reader still sees the original, unprefixed text.

This is the cheap, deterministic version of the technique. It costs nothing and helps,
but it is not the same as generating a bespoke context sentence per chunk with a
language model, and the guide does not claim the published gains from that heavier
method. An opt-in mode for the model-generated variant is left as a documented
extension rather than implied here.
"""

from __future__ import annotations

from .models import Chunk


def augment(chunks: list[Chunk]) -> list[Chunk]:
    total = len(chunks)
    for chunk in chunks:
        header_bits = [chunk.source]
        header_bits.extend(chunk.heading_path)
        header = " > ".join(bit for bit in header_bits if bit)
        position = f"part {chunk.ordinal + 1} of {total}"
        chunk.augmented_text = f"[{header} | {position}]\n{chunk.text}"
    return chunks
