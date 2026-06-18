from app.fusion import fuse
from app.models import Chunk


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, text=cid, source="doc", ordinal=int(cid[-1]))


def test_fusion_rewards_agreement_between_retrievers():
    a, b, c = _chunk("c0"), _chunk("c1"), _chunk("c2")
    dense = [(a, 0.9), (b, 0.8), (c, 0.7)]
    sparse = [(a, 5.0), (c, 3.0), (b, 1.0)]
    fused = fuse(dense, sparse, k=60)
    # a is top in both lists, so it must win
    assert fused[0].chunk.id == "c0"
    assert fused[0].dense_rank == 1 and fused[0].sparse_rank == 1


def test_fusion_includes_items_found_by_only_one_retriever():
    a, b = _chunk("c0"), _chunk("c1")
    fused = fuse([(a, 0.9)], [(b, 2.0)], k=60)
    ids = {c.chunk.id for c in fused}
    assert ids == {"c0", "c1"}
    only_dense = next(c for c in fused if c.chunk.id == "c0")
    assert only_dense.sparse_rank is None and only_dense.dense_rank == 1


def test_fusion_score_matches_formula():
    a = _chunk("c0")
    fused = fuse([(a, 0.5)], [(a, 0.5)], k=60)
    assert abs(fused[0].rrf_score - (1 / 61 + 1 / 61)) < 1e-9
