"""Construye el índice a partir de los documentos en data/docs.

Uso:
    python ingest.py
"""

from rag.pipeline import DOCS_DIR, INDEX_PATH, build_index

if __name__ == "__main__":
    retriever = build_index()
    docs = {c.doc for c in retriever.chunks}
    print(f"Documentos: {len(docs)} | Fragmentos: {len(retriever.chunks)} | "
          f"Vocabulario: {len(retriever.vectorizer.vocabulary_)} términos")
    print(f"Índice guardado en {INDEX_PATH.relative_to(DOCS_DIR.parent.parent)}")
