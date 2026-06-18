from eval import metrics


def test_hit_rate_counts_any_relevant_in_top_k():
    ranked = ["a", "b", "c", "d"]
    assert metrics.hit_rate_at_k(ranked, {"c"}, 3) == 1.0
    assert metrics.hit_rate_at_k(ranked, {"d"}, 3) == 0.0
    assert metrics.hit_rate_at_k(ranked, {"d"}, 5) == 1.0


def test_recall_is_fraction_of_relevant_found():
    ranked = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(ranked, {"a", "c"}, 3) == 1.0
    assert metrics.recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert metrics.recall_at_k(ranked, set(), 3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert metrics.mrr(["a", "b", "c"], {"a"}) == 1.0
    assert metrics.mrr(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_is_one_when_relevant_ranked_first():
    assert metrics.ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0
    # a single relevant item lower down scores less than one
    assert metrics.ndcg_at_k(["a", "b"], {"b"}, 2) < 1.0
    assert metrics.ndcg_at_k(["a", "b"], set(), 2) == 0.0
