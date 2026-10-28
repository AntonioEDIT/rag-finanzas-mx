"""Chat por consola con el RAG.

Ejemplos:
    python cli.py --provider groq
    python cli.py --provider claude -q "¿Qué es el CAT?"
    python cli.py --provider none        # solo búsqueda, sin LLM
"""

import argparse

from dotenv import load_dotenv

from rag.pipeline import RAG


def print_answer(res) -> None:
    print(f"\n🤖 ({res.provider}) {res.answer}\n")
    if res.sources:
        print("📚 Fuentes:")
        for i, (c, score) in enumerate(res.sources, start=1):
            print(f"  [{i}] {c.title} › {c.section}  ({c.doc}, score={score:.2f})")
        if res.provider == "none":
            print("\n--- Fragmento principal ---")
            print(res.sources[0][0].text)
    print()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAG de finanzas personales en México")
    parser.add_argument("--provider", default="none", choices=["claude", "groq", "none"])
    parser.add_argument("-k", type=int, default=4, help="Fragmentos a recuperar")
    parser.add_argument("-q", "--question", help="Pregunta única (sin modo interactivo)")
    args = parser.parse_args()

    rag = RAG(provider=args.provider, k=args.k)

    if args.question:
        print_answer(rag.ask(args.question))
        return

    print("💬 Pregunta sobre finanzas personales y banca en México. Escribe 'salir' para terminar.")
    while True:
        q = input("Tú: ").strip()
        if q.lower() in {"salir", "exit", "quit"}:
            break
        if q:
            print_answer(rag.ask(q))


if __name__ == "__main__":
    main()
