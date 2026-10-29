"""Interfaz web con Streamlit.

Uso:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RAG

load_dotenv()

st.set_page_config(page_title="RAG Finanzas MX", page_icon="💰", layout="centered")

PROVIDERS = {
    "Groq · gpt-oss (gratis)": "groq",
    "Claude · Anthropic": "claude",
    "Sin LLM (solo búsqueda)": "none",
}

with st.sidebar:
    st.header("⚙️ Configuración")
    label = st.radio("Modelo", list(PROVIDERS.keys()))
    provider = PROVIDERS[label]
    k = st.slider("Fragmentos a recuperar (k)", 1, 8, 4)

    needed = {"groq": "GROQ_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(provider)
    if needed and not os.getenv(needed):
        st.warning(f"Falta `{needed}` en tu archivo .env")

    st.divider()
    st.caption("Base de conocimiento: tarjetas de crédito, CAT, Buró de Crédito, "
               "CETES, IPAB, Afore, fraudes, CONDUSEF y presupuesto.")
    st.caption("⚠️ Proyecto educativo. No es asesoría financiera.")


@st.cache_resource(show_spinner=False)
def get_rag(provider: str, k: int) -> RAG:
    return RAG(provider=provider, k=k)


st.title("💰 Asistente de Finanzas Personales MX")
st.write("Pregunta sobre crédito, ahorro y banca en México. "
         "Las respuestas se basan solo en la base de conocimiento del proyecto.")

examples = ["¿Qué diferencia hay entre el pago mínimo y el pago para no generar intereses?",
            "¿Cuánto protege el IPAB mis ahorros?",
            "¿Qué datos nunca me va a pedir mi banco?"]
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["pending"] = ex

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Escribe tu pregunta…") or st.session_state.pop("pending", None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando en la base de conocimiento…"):
                res = get_rag(provider, k).ask(question)
            text = res.answer
            if res.sources:
                refs = "\n".join(f"- **[{i}]** {c.title} › {c.section} · `{c.doc}` · score {s:.2f}"
                                 for i, (c, s) in enumerate(res.sources, 1))
                text += f"\n\n**Fuentes**\n{refs}"
            st.markdown(text)
            if res.sources:
                with st.expander("Ver fragmentos recuperados"):
                    for i, (c, s) in enumerate(res.sources, 1):
                        st.markdown(f"**[{i}] {c.section}** — {c.text}")
        except Exception as e:  # muestra errores de API de forma amigable
            text = f"❌ Error: {e}"
            st.error(text)
    st.session_state.history.append({"role": "assistant", "content": text})
