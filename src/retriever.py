"""
Lightweight retrieval module for the medical knowledge base.

This implements the "R" (Retrieval) in Retrieval-Augmented Generation (RAG)
using classic TF-IDF + cosine similarity. This keeps the project dependency
footprint small (no large embedding models / vector DB required) while
still demonstrating the standard RAG retrieval pattern. In a production
system this module could be swapped for dense embeddings (e.g.
sentence-transformers) with a vector store (FAISS / Chroma) without
changing the rest of the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KBEntry:
    id: str
    condition: str
    category: str
    symptoms: str
    advice: str
    red_flag: bool

    def as_context(self) -> str:
        return (
            f"Condition: {self.condition}\n"
            f"Category: {self.category}\n"
            f"Typical symptoms: {self.symptoms}\n"
            f"General guidance: {self.advice}"
        )


class MedicalKBRetriever:
    """TF-IDF based retriever over the curated medical knowledge base."""

    def __init__(self, kb_path: str | Path):
        self.kb_path = Path(kb_path)
        self.entries: List[KBEntry] = self._load_kb(self.kb_path)

        # Build the searchable corpus from condition name + symptoms text
        # ONLY (not the advice text). A user's query is describing symptoms,
        # so matching against symptom wording is far more precise than also
        # matching against advice/instructional text, which tends to share
        # generic action words (e.g. "call", "immediately", "time") across
        # many entries and can otherwise pull in unrelated conditions.
        # The advice text is still shown to the LLM/user via as_context().
        self._corpus = [
            f"{e.condition}. {e.symptoms}" for e in self.entries
        ]
        # ngram_range=(1, 2) so that phrases like "trouble speaking" are
        # distinct features from "trouble sleeping" - with a small corpus,
        # unigram-only TF-IDF can give a single coincidentally shared rare
        # word (e.g. "trouble") too much weight and cause false-positive
        # matches, which matters a lot for a safety-relevant retriever.
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    @staticmethod
    def _load_kb(path: Path) -> List[KBEntry]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [KBEntry(**item) for item in raw]

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.05,
        relative_cutoff: float = 0.75,
    ) -> List[KBEntry]:
        """
        Return the top_k most relevant KB entries for the query.

        Two filters are applied together, which matters for a small corpus
        like this one where TF-IDF can occasionally give a single shared
        word a high score:
          - min_score: an absolute floor below which a match is considered
            noise.
          - relative_cutoff: entries scoring below `relative_cutoff` times
            the top match's score are dropped, so a weak, tangential match
            doesn't get pulled in just to fill out top_k alongside a much
            stronger match.
        """
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_idx = scores.argsort()[::-1]

        top_score = scores[ranked_idx[0]] if len(ranked_idx) else 0.0
        dynamic_floor = max(min_score, top_score * relative_cutoff)

        results = []
        for idx in ranked_idx[:top_k]:
            if scores[idx] >= dynamic_floor and scores[idx] >= min_score:
                results.append(self.entries[idx])
        return results

    def get_red_flag_entries(self) -> List[KBEntry]:
        return [e for e in self.entries if e.red_flag]
