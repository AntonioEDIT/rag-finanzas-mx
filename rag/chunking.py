"""Carga de documentos Markdown y división en fragmentos (chunks).

Estrategia: cada documento se divide por secciones (encabezados ##).
Si una sección es muy larga, se parte en ventanas con traslape.
Cada chunk conserva el título del documento y de la sección para
dar contexto al buscador y para citar la fuente en la respuesta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Chunk:
    id: str
    doc: str        # nombre del archivo
    title: str      # título del documento (# ...)
    section: str    # título de la sección (## ...)
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def searchable_text(self) -> str:
        """Texto que se indexa: incluye títulos para mejorar la búsqueda."""
        return f"{self.title}. {self.section}. {self.text}"


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # intenta cortar en un salto de línea o punto para no partir frases
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_markdown(path: Path, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    # separa por encabezados de nivel 2
    parts = re.split(r"^##\s+(.+)$", raw, flags=re.MULTILINE)
    # parts = [intro, sec1_title, sec1_body, sec2_title, sec2_body, ...]
    sections: list[tuple[str, str]] = []
    intro = re.sub(r"^#\s+.+$", "", parts[0], flags=re.MULTILINE).strip()
    if intro:
        sections.append(("Introducción", intro))
    for i in range(1, len(parts), 2):
        sections.append((parts[i].strip(), parts[i + 1].strip()))

    chunks: list[Chunk] = []
    for s_idx, (section, body) in enumerate(sections):
        for p_idx, piece in enumerate(_split_long(body, max_chars, overlap)):
            chunks.append(
                Chunk(
                    id=f"{path.stem}#{s_idx}.{p_idx}",
                    doc=path.name,
                    title=title,
                    section=section,
                    text=piece,
                )
            )
    return chunks


def load_corpus(docs_dir: Path, **kwargs) -> list[Chunk]:
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No hay archivos .md en {docs_dir}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_markdown(f, **kwargs))
    return chunks
