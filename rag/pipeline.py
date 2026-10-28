"""Pipeline RAG: recuperar fragmentos -> construir prompt -> generar respuesta."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, load_corpus
from .llm import LLMProvider, get_provider
from .retriever import TfidfRetriever

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_PATH = ROOT / "data" / "index" / "tfidf.joblib"

SYSTEM_PROMPT = """Eres un asistente de educación financiera para usuarios en México.
Respondes en español, de forma clara y breve.

Reglas:
1. Usa ÚNICAMENTE la información del CONTEXTO. No inventes datos, cifras ni leyes.
2. Cita las fuentes con su número entre corchetes, por ejemplo [1] o [2][3].
3. Si el contexto no contiene la respuesta, dilo claramente: "No encontré esa información en la base de conocimiento".
4. No des recomendaciones de inversión personalizadas; ofrece información general.
"""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[tuple[Chunk, float]] = field(default_factory=list)
    provider: str = "none"


def build_index(docs_dir: Path = DOCS_DIR, index_path: Path = INDEX_PATH) -> TfidfRetriever:
    chunks = load_corpus(docs_dir)
    retriever = TfidfRetriever(chunks)
    retriever.save(index_path)
    return retriever


def load_or_build(index_path: Path = INDEX_PATH) -> TfidfRetriever:
    if index_path.exists():
        return TfidfRetriever.load(index_path)
    return build_index(index_path=index_path)


def format_context(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for i, (c, _score) in enumerate(results, start=1):
        blocks.append(f"[{i}] {c.title} › {c.section}\n{c.text}")
    return "\n\n".join(blocks)


class RAG:
    def __init__(self, provider: LLMProvider | str = "none", k: int = 4,
                 retriever: TfidfRetriever | None = None):
        self.retriever = retriever or load_or_build()
        self.provider = get_provider(provider) if isinstance(provider, str) else provider
        self.k = k

    def ask(self, question: str) -> RAGAnswer:
        results = self.retriever.search(question, k=self.k)
        if not results:
            return RAGAnswer(question, "No encontré esa información en la base de conocimiento.",
                             [], self.provider.name)

        user_prompt = (f"CONTEXTO:\n{format_context(results)}\n\n"
                       f"PREGUNTA: {question}\n\nRespuesta:")
        answer = self.provider.generate(SYSTEM_PROMPT, user_prompt)
        return RAGAnswer(question, answer, results, self.provider.name)
