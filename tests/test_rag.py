"""Pruebas del retriever: cada pregunta debe recuperar el documento correcto en el top 1."""

import pytest

from rag.chunking import load_corpus
from rag.pipeline import DOCS_DIR, RAG
from rag.retriever import TfidfRetriever


@pytest.fixture(scope="module")
def retriever():
    return TfidfRetriever(load_corpus(DOCS_DIR))


CASOS = [
    ("¿Qué pasa si solo pago el mínimo de mi tarjeta?", "01_tarjetas_de_credito.md"),
    ("¿Qué es el Costo Anual Total?", "02_cat_y_tasas_de_interes.md"),
    ("¿Cada cuánto puedo pedir mi reporte de crédito gratis?", "03_historial_crediticio.md"),
    ("¿A qué plazos puedo comprar CETES?", "04_ahorro_e_inversion.md"),
    ("¿Cuántas UDIs protege el IPAB?", "05_ipab_proteccion_al_ahorro.md"),
    ("¿Cómo hago aportaciones voluntarias a mi Afore?", "06_afore_y_retiro.md"),
    ("Me llamaron del banco pidiendo el código de mi token", "07_fraudes_bancarios.md"),
    ("¿Dónde pongo una queja contra un banco?", "08_condusef_y_derechos.md"),
    ("¿Qué es la regla 50/30/20?", "09_presupuesto_personal.md"),
]


@pytest.mark.parametrize("pregunta,doc_esperado", CASOS)
def test_top1(retriever, pregunta, doc_esperado):
    resultados = retriever.search(pregunta, k=3)
    assert resultados, "No se recuperó nada"
    assert resultados[0][0].doc == doc_esperado


def test_pregunta_fuera_de_tema(retriever):
    assert retriever.search("receta de pozole rojo", k=3) == []


def test_pipeline_sin_llm():
    res = RAG(provider="none").ask("¿Qué es el CAT?")
    assert res.sources and res.provider == "none"
