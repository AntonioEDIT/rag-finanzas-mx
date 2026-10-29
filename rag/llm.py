"""Proveedores de LLM para la etapa de generación.

- claude: API de Anthropic (de pago, requiere ANTHROPIC_API_KEY).
- groq:   API de Groq con modelos abiertos como Llama (tiene plan gratuito,
          requiere GROQ_API_KEY).
- none:   sin LLM; devuelve los fragmentos recuperados. Útil para probar
          el buscador sin ninguna llave.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str | None = None, max_tokens: int = 800):
        from anthropic import Anthropic  # import perezoso
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Falta ANTHROPIC_API_KEY en el archivo .env")
        self.client = Anthropic()
        self.model = model or os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.2,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, model: str | None = None, max_tokens: int = 2000):
        from groq import Groq  # import perezoso
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Falta GROQ_API_KEY en el archivo .env")
        self.client = Groq()
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class NoLLMProvider(LLMProvider):
    """Modo demo: no genera, solo indica que se muestran los fragmentos."""
    name = "none"

    def generate(self, system: str, user: str) -> str:
        return ("(Modo sin LLM) Estos son los fragmentos más relevantes encontrados. "
                "Configura ANTHROPIC_API_KEY o GROQ_API_KEY para obtener una respuesta redactada.")


def get_provider(name: str, model: str | None = None) -> LLMProvider:
    name = (name or "none").lower()
    if name == "claude":
        return ClaudeProvider(model)
    if name == "groq":
        return GroqProvider(model)
    if name == "none":
        return NoLLMProvider()
    raise ValueError(f"Proveedor desconocido: {name}. Usa claude, groq o none.")
