from __future__ import annotations

import sys
from pathlib import Path
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st

try:
    from streamlit_elements import elements, mui, nivo
except ImportError:  # pragma: no cover - elemento visual opcional
    elements = mui = nivo = None

# Garante acesso ao módulo chunks.py mesmo executando o app dentro de chunks_lab
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunks import Chunks


st.set_page_config(
    page_title="Explorando Chunks",
    page_icon="✂️",
    layout="wide",
)

st.title("👩‍🏫 Visualizando o funcionamento da classe Chunks")
st.markdown(
    """
Este laboratório interativo demonstra como o algoritmo de fragmentação trabalha. Ajuste os parâmetros
na barra lateral, forneça um texto e veja como os pedaços são gerados, além dos metadados calculados.
    """
)


with st.sidebar:
    st.header("Configurações")
    st.markdown(
        "Ajuste o tamanho dos pedaços e a sobreposição para analisar o impacto no resultado."
    )

    default_text = (
        "Os drones estão revolucionando diversas indústrias, desde o monitoramento "
        "ambiental até entregas de última milha.\n\n"
        "Entretanto, o uso massivo desses dispositivos exige novas soluções de detecção "
        "para garantir segurança e privacidade.\n\n"
        "Métodos tradicionais não acompanham a velocidade dessa evolução, tornando essencial "
        "o uso de abordagens modernas."
    )

    st.subheader("Entrada de texto")
    st.badge("Texto manual", icon=":material/edit:", color="violet")
    input_text = st.text_area(
        "Cole ou digite o conteúdo a ser fragmentado",
        value=default_text,
        height=220,
    )

    uploaded_file = st.file_uploader(
        "Ou envie um arquivo (.pdf, .md, .txt)",
        type=["pdf", "md", "markdown", "txt"],
    )
    st.badge("Upload opcional", icon=":material/upload:", color="blue")
    if uploaded_file is not None:
        st.success(f"Arquivo carregado: {uploaded_file.name}")
    else:
        st.caption("Sem arquivo enviado - será usado o texto manual.")

    st.subheader("Parâmetros")
    chunk_size = st.slider("chunk_size", min_value=40, max_value=600, value=180, step=10)
    overlap_size = st.slider(
        "overlap_size",
        min_value=0,
        max_value=max(chunk_size - 10, 0),
        value=min(30, chunk_size - 10),
        step=5,
    )

    st.subheader("Execução")
    run_button = st.button("Gerar chunks", width='content')


def extract_text_from_upload(upload) -> Tuple[str, Optional[bytes]]:
    """Retorna texto normalizado e, quando aplicável, os bytes originais do PDF."""

    if upload is None:
        return "", None

    filename = upload.name.lower()
    raw_bytes = upload.getvalue()

    if filename.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            st.sidebar.error(
                "Para ler PDFs, instale a dependência PyPDF2 (ex.: `pip install PyPDF2`)."
            )
            raise RuntimeError("Dependência PyPDF2 ausente") from exc

        text_parts: List[str] = []
        try:
            reader = PdfReader(BytesIO(raw_bytes))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        except Exception as exc:
            st.sidebar.error(f"Não foi possível extrair texto do PDF: {exc}")
            raise RuntimeError("Falha ao ler PDF") from exc

        return "\n\n".join(text_parts), raw_bytes

    decoded: str
    try:
        decoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        decoded = raw_bytes.decode("latin-1", errors="ignore")

    return decoded, None


def build_chunks_dataframe(chunks: List[str]) -> pd.DataFrame:
    """Converte a lista de strings em DataFrame pronto para visualização."""

    return pd.DataFrame(
        {
            "chunk_index": range(len(chunks)),
            "chunk_text": chunks,
            "chunk_length": [len(c) for c in chunks],
        }
    )


def build_metadata_dataframe(metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepara DataFrame com colunas normalizadas do metadata."""

    if not metadata:
        return pd.DataFrame(columns=["chunk_id", "chunk_size", "chunk_start_char"])

    normalized = [
        {
            "chunk_id": entry.get("chunk_id"),
            "chunk_size": entry.get("chunk_size"),
            "chunk_start_char": entry.get("chunk_start_char"),
            "total_chunks": entry.get("total_chunks"),
            "preview": entry.get("chunk_text", "")[:80] + "...",
        }
        for entry in metadata
    ]
    return pd.DataFrame(normalized)


uploaded_text: str = ""
pdf_bytes: Optional[bytes] = None
upload_error: Optional[str] = None

if uploaded_file is not None:
    try:
        uploaded_text, pdf_bytes = extract_text_from_upload(uploaded_file)
    except RuntimeError as exc:
        upload_error = str(exc)


if run_button:
    try:
        if upload_error:
            st.error(upload_error)
            st.stop()

        source_text = uploaded_text.strip() if uploaded_text.strip() else input_text
        if not source_text.strip():
            st.warning("Forneça um texto na área ou envie um arquivo válido.")
            st.stop()

        source_info: Dict[str, Any] = {"cenario": "demo_streamlit"}
        if uploaded_file is not None:
            source_info.update(
                {
                    "arquivo": uploaded_file.name,
                    "tamanho_bytes": len(uploaded_file.getvalue()),
                }
            )

        chunker = Chunks(chunk_size=chunk_size, overlap_size=overlap_size)
        chunk_list = chunker.create_chunks(source_text)
        metadata_list = chunker.create_chunks_with_metadata(
            source_text,
            source_info=source_info,
        )
        info = chunker.get_chunk_info()

        st.toast("Chunks atualizados!", icon="✅")
        if chunk_list:
            st.balloons()

        col_total, col_step, col_overlap = st.columns(3)
        col_total.metric("Total de chunks", len(chunk_list))
        col_step.metric("Passo efetivo", info["effective_chunk_step"])
        col_overlap.metric("Overlap", info["overlap_size"])

        st.divider()

        if elements is not None and mui is not None:
            elements_frame = cast(Any, elements)
            mui_module = cast(Any, mui)
            nivo_module = cast(Any, nivo) if nivo is not None else None

            pipeline_steps = [
                {"label": "Entrada", "info": f"{len(source_text)} caracteres"},
                {"label": "Pré-processamento", "info": f"{len(chunk_list)} chunks"},
                {"label": "Metadados", "info": f"{len(metadata_list)} registros"},
            ]

            bar_data = [
                {"chunk": f"Chunk {meta['chunk_id']}", "tamanho": meta["chunk_size"]}
                for meta in metadata_list
            ]

            with elements_frame("chunks_visual"):
                with mui_module.Stack(spacing=3):
                    with mui_module.Card(
                        elevation=4,
                        sx={
                            "padding": 24,
                            "backgroundColor": "#12161f",
                            "color": "#f4f6fb",
                            "borderRadius": 4,
                        },
                    ):
                        mui_module.Typography(
                            "Fluxo de processamento", variant="h6", gutterBottom=True
                        )
                        with mui_module.Stepper(activeStep=2, alternativeLabel=True):
                            for step in pipeline_steps:
                                with mui_module.Step():
                                    mui_module.StepLabel(step["label"])
                        with mui_module.Stack(direction="row", spacing=1, sx={"mt": 2}):
                            for step in pipeline_steps:
                                mui_module.Chip(
                                    label=step["info"],
                                    color="primary",
                                    variant="outlined",
                                )

                    with mui_module.Card(
                        elevation=4,
                        sx={
                            "padding": 24,
                            "backgroundColor": "#1a2130",
                            "color": "#f4f6fb",
                            "borderRadius": 4,
                        },
                    ):
                        mui_module.Typography(
                            "Distribuição do tamanho dos chunks",
                            variant="h6",
                            gutterBottom=True,
                        )
                        if bar_data and nivo_module is not None:
                            with mui_module.Box(sx={"height": 320}):
                                nivo_module.Bar(
                                    data=bar_data,
                                    keys=["tamanho"],
                                    indexBy="chunk",
                                    margin={"top": 20, "right": 20, "bottom": 60, "left": 60},
                                    padding=0.3,
                                    colors={"scheme": "set2"},
                                    enableLabel=True,
                                    labelTextColor="#000",
                                    axisBottom={
                                        "tickRotation": -35,
                                        "tickPadding": 14,
                                        "legend": "Chunks",
                                        "legendOffset": 45,
                                    },
                                    axisLeft={"legend": "Caracteres", "legendOffset": -50},
                                    animate=True,
                                )
                        else:
                            mui_module.Alert(
                                "Gráfico indisponível — instale o extra Nivo ou gere chunks para visualizar.",
                                severity="warning",
                            )

                    with mui_module.Card(
                        elevation=4,
                        sx={
                            "padding": 24,
                            "backgroundColor": "#12161f",
                            "color": "#f4f6fb",
                            "borderRadius": 4,
                        },
                    ):
                        mui_module.Typography(
                            "Detalhes de cada chunk", variant="h6", gutterBottom=True
                        )
                        if not metadata_list:
                            mui_module.Typography(
                                "Nenhum chunk foi criado.", variant="body2"
                            )
                        else:
                            with mui_module.Stack(spacing=2):
                                for meta in metadata_list[:10]:
                                    with mui_module.Accordion(defaultExpanded=meta["chunk_id"] == 0):
                                        expand_icon = (
                                            mui_module.icon.ExpandMore()
                                            if hasattr(mui_module.icon, "ExpandMore")
                                            else None
                                        )
                                        with mui_module.AccordionSummary(expandIcon=expand_icon):
                                            mui_module.Typography(
                                                f"Chunk {meta['chunk_id']} • {meta['chunk_size']} caracteres",
                                                sx={"fontWeight": 600},
                                            )
                                        with mui_module.AccordionDetails():
                                            mui_module.Typography(
                                                meta["chunk_text"], variant="body2"
                                            )
                                            progresso = 0.0
                                            if chunk_size:
                                                progresso = min(
                                                    meta["chunk_size"] / float(chunk_size) * 100.0,
                                                    100.0,
                                                )
                                            mui_module.LinearProgress(
                                                variant="determinate",
                                                value=progresso,
                                                sx={"mt": 2},
                                            )
                                            with mui_module.Stack(
                                                direction="row",
                                                spacing=2,
                                                sx={"mt": 1},
                                            ):
                                                mui_module.Chip(
                                                    label=f"Início estimado: {meta['chunk_start_char']}"
                                                )
                                                mui_module.Chip(
                                                    label=f"Total de chunks: {meta['total_chunks']}"
                                                )
        else:
            st.info(
                "Instale `streamlit-elements==0.1.*` para visualizar o painel didático complementar.",
                icon="ℹ️",
            )

        tabs = st.tabs(["Chunks", "Metadados", "Configuração"])

        with tabs[0]:
            st.subheader("Chunks gerados")
            st.badge("Visão dos blocos", icon=":material/view_week:", color="primary")
            chunk_df = build_chunks_dataframe(chunk_list)
            if chunk_df.empty:
                st.info("Nenhum chunk foi criado. Ajuste o texto ou os parâmetros.")
            else:
                st.dataframe(chunk_df, width='content', hide_index=True)
                with st.expander("Ver texto completo de cada chunk"):
                    for idx, chunk_text in enumerate(chunk_list, start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.code(chunk_text)
                st.caption(
                    "Dica: destaque os trechos onde o algoritmo escolheu cortes naturais para discutir com a turma."
                )

        with tabs[1]:
            st.subheader("Metadados dos chunks")
            st.badge("Detalhes analíticos", icon=":material/insights:", color="green")
            metadata_df = build_metadata_dataframe(metadata_list)
            if metadata_df.empty:
                st.warning("Metadados indisponíveis.")
            else:
                st.dataframe(metadata_df, width='content', hide_index=True)
                st.markdown("### Estrutura completa")
                st.json(metadata_list)
            if pdf_bytes:
                st.markdown("### Visualização do PDF")
                try:
                    st.pdf(BytesIO(pdf_bytes))
                except Exception as exc:
                    st.info(
                        "Para visualizar PDFs diretamente, instale o extra `streamlit[pdf]`."
                    )
                    st.caption(f"Detalhes do erro: {exc}")

        with tabs[2]:
            st.subheader("Como o algoritmo funciona")
            st.badge("Passo a passo", icon=":material/auto_stories:", color="violet")
            st.markdown(
                "A classe `Chunks` tenta respeitar limites naturais do texto, priorizando parágrafos, "
                "frases e espaços para cortes."
            )
            with st.echo("below"):
                demo_chunker = Chunks(chunk_size=120, overlap_size=30)
                demo_chunks = demo_chunker.create_chunks("Texto de exemplo.")
                st.write(demo_chunks)

            st.markdown("### Configuração atual")
            st.json(info)

    except ValueError as exc:
        st.error(f"Erro ao gerar chunks: {exc}")
else:
    st.info("Defina suas preferências na barra lateral e clique em 'Gerar chunks'.")