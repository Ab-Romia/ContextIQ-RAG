"""TF-IDF baseline retriever.

This is the "before" arm of the ablation. It is a correct, textbook TF-IDF retriever:
fit on the whole corpus once, cosine similarity against the query vector. That makes the
comparison fair. The original project's flaw was not that it used TF-IDF, it was that it
used TF-IDF wrongly (a vectorizer fit once and frozen, with a hash fallback). Comparing
the new pipeline against a properly built TF-IDF baseline is the honest test: it shows
the gain comes from semantic and hybrid retrieval, not from fixing a bug.

scikit-learn is an evaluation-only dependency (see eval/requirements.txt); it is not in
the application image.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Chunk


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    def rank(self, query: str, k: int) -> list[str]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [self._chunks[i].id for i in order]
