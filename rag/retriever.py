"""Buscador (retriever) basado en TF-IDF de n-gramas de caracteres + similitud coseno.

Se eligió TF-IDF porque:
- No necesita descargar modelos ni GPU: corre en cualquier laptop.
- Es rápido y explicable.
- Para un corpus pequeño y especializado funciona bien.

Se usan n-gramas de caracteres (3 a 5) en lugar de palabras completas porque
toleran plurales, conjugaciones y faltas de ortografía: "queja" encuentra
"quejas" y "credito" encuentra "crédito". Con TF-IDF por palabras, la
pregunta "¿Dónde pongo una queja?" no encontraba el documento de CONDUSEF.

La clase expone la misma interfaz (`search`) que tendría un buscador con
embeddings, así que se puede cambiar después sin tocar el resto.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk

# Por debajo de este puntaje se considera que la pregunta no tiene relación
# con la base de conocimiento (calibrado con preguntas dentro y fuera de tema).
DEFAULT_MIN_SCORE = 0.12


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",        # n-gramas de caracteres dentro de cada palabra
            ngram_range=(3, 5),
            strip_accents="unicode",   # "crédito" == "credito"
            lowercase=True,
            sublinear_tf=True,
        )
        self.chunks: list[Chunk] = []
        self.matrix = None
        if chunks:
            self.fit(chunks)

    def fit(self, chunks: list[Chunk]) -> "TfidfRetriever":
        self.chunks = chunks
        self.matrix = self.vectorizer.fit_transform([c.searchable_text for c in chunks])
        return self

    def search(self, query: str, k: int = 4,
               min_score: float = DEFAULT_MIN_SCORE) -> list[tuple[Chunk, float]]:
        if self.matrix is None:
            raise RuntimeError("El índice está vacío. Ejecuta primero la ingesta.")
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]

    # --- persistencia -------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "chunks": [c.to_dict() for c in self.chunks]}, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        data = joblib.load(path)
        r = cls()
        r.vectorizer = data["vectorizer"]
        r.matrix = data["matrix"]
        r.chunks = [Chunk(**d) for d in data["chunks"]]
        return r
