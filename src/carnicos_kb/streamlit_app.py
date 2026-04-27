"""Interfaz web Streamlit para el sistema Q&A de Alimentos Carnicos."""

import os
from pathlib import Path
from typing import Dict, Optional

import streamlit as st
from dotenv import load_dotenv

from carnicos_kb.knowledge_loader import get_knowledge_stats
from carnicos_kb.paths import DEFAULT_CHUNKS_FILE, DEFAULT_DATASET_DIR
from carnicos_kb.qa_system import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    CarnicosQASystem,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

EXAMPLE_QUESTIONS = [
    "Que empresa es Alimentos Carnicos S.A.S.?",
    "Que marcas hacen parte del portafolio?",
    "Como puedo radicar una PQRS?",
    "Donde esta ubicada la sede principal?",
    "Que compromisos existen sobre bienestar animal?",
    "Que temas estan fuera del alcance del asistente?",
]

MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ðŸ")


def main() -> None:
    st.set_page_config(
        page_title="Asistente Q&A | Alimentos Carnicos",
        page_icon="AC",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    config = render_sidebar()

    st.title("Asistente Q&A - Alimentos Carnicos")
    st.caption(
        "Consulta la base de conocimiento construida con scraping web, PDFs publicos "
        "y documentos del proyecto."
    )

    assistant_tab, scope_tab, report_tab, guide_tab, knowledge_tab = st.tabs(
        [
            "Asistente",
            "Definicion del alcance",
            "Informes",
            "Guia rapida",
            "Base de conocimiento",
        ]
    )

    with assistant_tab:
        render_assistant(config)

    with scope_tab:
        render_scope()

    with report_tab:
        render_reports()

    with guide_tab:
        render_quick_guide()

    with knowledge_tab:
        render_knowledge_base(config["knowledge_path"])


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        .small-note {
            color: #5f6b7a;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Dict[str, object]:
    st.sidebar.title("Configuracion")

    api_key_ready = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_ready:
        st.sidebar.success("OPENAI_API_KEY configurada")
    else:
        st.sidebar.warning("Falta configurar OPENAI_API_KEY")

    default_model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    default_temperature = parse_float(os.getenv("OPENAI_TEMPERATURE"), DEFAULT_TEMPERATURE)
    default_max_tokens = parse_int(os.getenv("OPENAI_MAX_TOKENS"), DEFAULT_MAX_TOKENS)

    model = st.sidebar.text_input("Modelo", value=default_model)
    temperature = st.sidebar.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=float(default_temperature),
        step=0.05,
        help="Valores bajos reducen variacion y favorecen respuestas verificables.",
    )
    max_tokens = st.sidebar.number_input(
        "Max tokens",
        min_value=200,
        max_value=4000,
        value=int(default_max_tokens),
        step=100,
    )

    knowledge_path = resolve_knowledge_path()
    if knowledge_path.exists():
        st.sidebar.info(f"Base: `{format_path(knowledge_path)}`")
    else:
        st.sidebar.error(f"No existe la base: `{format_path(knowledge_path)}`")

    if st.sidebar.button("Limpiar conversacion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.sidebar.button("Recargar asistente", use_container_width=True):
        get_qa_system.clear()
        st.session_state.messages = []
        st.rerun()

    return {
        "api_key_ready": api_key_ready,
        "knowledge_path": knowledge_path,
        "model": model.strip() or DEFAULT_MODEL,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }


def render_assistant(config: Dict[str, object]) -> None:
    st.subheader("Pregunta al modelo")
    st.write(
        "El asistente responde dentro del alcance definido y debe admitir cuando la "
        "informacion no exista en la base de conocimiento."
    )

    if not config["knowledge_path"].exists():
        st.error(
            "No se encontro la base de conocimiento. Ejecuta `make chunk` o configura "
            "`CARNICOS_KNOWLEDGE_PATH` en `.env`."
        )
        return

    if not config["api_key_ready"]:
        st.warning(
            "Configura `OPENAI_API_KEY` en `.env` para habilitar las respuestas del modelo. "
            "Mientras tanto puedes revisar el alcance, la guia rapida y la base cargada."
        )
        return

    try:
        qa_system = get_qa_system(
            str(config["knowledge_path"]),
            str(config["model"]),
            float(config["temperature"]),
            int(config["max_tokens"]),
        )
    except Exception as exc:
        st.error(f"No fue posible inicializar el asistente: {exc}")
        return

    stats = qa_system.stats
    metric_cols = st.columns(3)
    metric_cols[0].metric("Caracteres", f"{stats['total_characters']:,}")
    metric_cols[1].metric("Palabras", f"{stats['total_words']:,}")
    metric_cols[2].metric("Parrafos", f"{stats['total_paragraphs']:,}")

    render_examples()
    render_chat(qa_system)


def render_examples() -> None:
    with st.expander("Preguntas sugeridas", expanded=True):
        cols = st.columns(3)
        for index, question in enumerate(EXAMPLE_QUESTIONS):
            if cols[index % 3].button(question, use_container_width=True):
                st.session_state.pending_question = question


def render_chat(qa_system: CarnicosQASystem) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_question = st.chat_input("Escribe una pregunta sobre Alimentos Carnicos...")
    pending_question = st.session_state.pop("pending_question", None)
    question = pending_question or typed_question

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando la base de conocimiento y el modelo..."):
            answer = qa_system.answer(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_scope() -> None:
    st.markdown(read_markdown("docs/definicion_alcance.md"))


def render_reports() -> None:
    st.subheader("Informes del proyecto")

    report_files = sorted((PROJECT_ROOT / "docs").glob("informe_modulo_*.md"))
    if not report_files:
        st.warning("No se encontraron informes en `docs/informe_modulo_*.md`.")
        return

    labels = [report_path.name for report_path in report_files]
    default_index = labels.index("informe_modulo_2.md") if "informe_modulo_2.md" in labels else 0
    selected_label = st.selectbox("Selecciona un informe", labels, index=default_index)
    selected_path = next(path for path in report_files if path.name == selected_label)

    if selected_label != "informe_modulo_2.md" and not (PROJECT_ROOT / "docs/informe_modulo_2.md").exists():
        st.info(
            "`docs/informe_modulo_2.md` aun no existe. Se muestra el informe disponible."
        )

    # Leer contenido
    content = read_markdown(str(selected_path.relative_to(PROJECT_ROOT)))

    # 🔧 Ajuste correcto para tablas
    content = content.replace("<br>", "<br/>")

    # Renderizar permitiendo HTML
    st.markdown(content, unsafe_allow_html=True)


def render_quick_guide() -> None:
    st.subheader("Guia rapida de uso")
    st.markdown(
        """
        1. Configura `.env` con `OPENAI_API_KEY`.
        2. Instala dependencias con `make sync`.
        3. Verifica o regenera la base con `make chunk`.
        4. Ejecuta la interfaz con `make app`.
        5. Formula preguntas dentro del alcance documentado.
        """
    )

    with st.expander("Documento de guia rapida del proyecto", expanded=False):
        st.markdown(read_markdown("docs/GUIA_RAPIDA.md"))


def render_knowledge_base(knowledge_path: Path) -> None:
    st.subheader("Estado de la base de conocimiento")

    if not knowledge_path.exists():
        st.error(f"No se encontro `{format_path(knowledge_path)}`.")
        return

    content, source_label, total_size, modified_at = load_knowledge_preview(knowledge_path)
    stats = get_knowledge_stats(content)

    cols = st.columns(4)
    cols[0].metric("Fuente", source_label)
    cols[1].metric("Caracteres", f"{stats['total_characters']:,}")
    cols[2].metric("Palabras", f"{stats['total_words']:,}")
    cols[3].metric("Tamano", f"{total_size / 1024:.1f} KB")

    st.markdown(
        f'<p class="small-note">Ruta: <code>{format_path(knowledge_path)}</code> '
        f'| Modificado: {format_timestamp(modified_at)}</p>',
        unsafe_allow_html=True,
    )

    preview_length = min(len(content), 5000)
    with st.expander("Vista previa del contenido cargado", expanded=False):
        st.code(content[:preview_length], language="markdown")
        if len(content) > preview_length:
            st.caption("Vista previa truncada para mantener la pagina ligera.")


def load_knowledge_preview(knowledge_path: Path) -> tuple[str, str, int, float]:
    if knowledge_path.is_file():
        return (
            read_text_file(knowledge_path),
            knowledge_path.name,
            knowledge_path.stat().st_size,
            knowledge_path.stat().st_mtime,
        )

    md_files = sorted(knowledge_path.glob("*.md"))
    if not md_files:
        return "", "0 archivos .md", 0, knowledge_path.stat().st_mtime

    parts = []
    total_size = 0
    modified_at = 0.0
    for md_file in md_files:
        parts.append(f"--- Fuente: {md_file.name} ---\n{read_text_file(md_file)}")
        total_size += md_file.stat().st_size
        modified_at = max(modified_at, md_file.stat().st_mtime)

    return "\n\n".join(parts), f"{len(md_files)} archivos .md", total_size, modified_at


@st.cache_resource(show_spinner=False)
def get_qa_system(
    knowledge_path: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> CarnicosQASystem:
    return CarnicosQASystem(
        knowledge_dir=knowledge_path,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=False,
    )


@st.cache_data(show_spinner=False)
def read_markdown(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return f"> Documento no encontrado: `{relative_path}`"
    return repair_mojibake(read_text_file(path))


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text

    original_score = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    repaired_score = sum(repaired.count(marker) for marker in MOJIBAKE_MARKERS)
    return repaired if repaired_score < original_score else text


def resolve_knowledge_path() -> Path:
    env_path = os.getenv("CARNICOS_KNOWLEDGE_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    chunks_path = PROJECT_ROOT / DEFAULT_CHUNKS_FILE
    if chunks_path.exists():
        return chunks_path

    return PROJECT_ROOT / DEFAULT_DATASET_DIR


def format_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def format_timestamp(timestamp: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def parse_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def parse_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


if __name__ == "__main__":
    main()
