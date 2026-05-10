"""
Validacion tecnica del punto de partida del Modulo 1.

Este modulo comprueba que la base de conocimiento y el sistema Q&A inicial
esten listos antes de comenzar la transformacion del Modulo 2. La validacion
esta pensada para uso academico y de sustentacion: imprime un diagnostico
claro, no revela secretos del archivo `.env` y distingue entre validaciones
locales y prueba real contra el LLM.

Alcance de la validacion:

1. Verifica que exista el corpus Markdown procesado.
2. Verifica que exista el archivo consolidado de chunks.
3. Carga la base de conocimiento usando el mismo loader de la aplicacion.
4. Revisa si hay una API key real configurada sin imprimirla.
5. Si hay API key, ejecuta una pregunta de prueba con `CarnicosQASystem`.

Nota metodologica:
El Modulo 1 de este proyecto no implementa una base vectorial ni embeddings en
tiempo de consulta. La arquitectura actual usa una base segmentada en chunks y
la inyecta completa en el prompt del sistema. Esta validacion confirma que ese
punto de partida funciona y queda listo para evolucionar a un agente con RAG,
memoria y herramientas en el Modulo 2.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from carnicos_kb.knowledge_loader import get_knowledge_stats, load_knowledge_base
from carnicos_kb.paths import DEFAULT_CHUNKS_FILE, DEFAULT_DATASET_DIR
from carnicos_kb.qa_system import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    CarnicosQASystem,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_QUESTION = "Que empresa es Alimentos Carnicos S.A.S.?"
CHUNK_HEADING_RE = re.compile(r"^## C\d{4}\s+\|", re.MULTILINE)


@dataclass(frozen=True)
class ValidationResult:
    """Resultado agregado de la validacion."""

    local_checks_ok: bool
    llm_check_ok: bool | None
    require_llm: bool

    @property
    def exit_code(self) -> int:
        if not self.local_checks_ok:
            return 1
        if self.require_llm and self.llm_check_ok is not True:
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida el punto de partida del Modulo 1: corpus procesado, "
            "archivo de chunks, carga de conocimiento y prueba opcional del LLM."
        )
    )
    parser.add_argument(
        "--knowledge-path",
        type=Path,
        default=None,
        help=(
            "Archivo o carpeta de conocimiento. Si se omite, usa "
            "CARNICOS_KNOWLEDGE_PATH o data/processed/base_conocimiento_chunks.md."
        ),
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_TEST_QUESTION,
        help="Pregunta usada para la prueba end-to-end contra el LLM.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Omite la llamada al LLM aunque exista OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Falla la validacion si no se puede ejecutar la prueba real del LLM.",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada de linea de comandos."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = parse_args()

    print_title()
    knowledge_path = resolve_knowledge_path(args.knowledge_path)
    local_checks_ok = run_local_checks(knowledge_path)
    llm_check_ok = run_llm_check(knowledge_path, args)

    result = ValidationResult(
        local_checks_ok=local_checks_ok,
        llm_check_ok=llm_check_ok,
        require_llm=bool(args.require_llm),
    )
    print_summary(result)
    raise SystemExit(result.exit_code)


def print_title() -> None:
    print("=" * 72)
    print("VALIDACION MODULO 1 - BASE Q&A / PRE-RAG")
    print("=" * 72)


def resolve_knowledge_path(argument_path: Path | None) -> Path:
    if argument_path is not None:
        return absolute_project_path(argument_path)

    env_path = os.getenv("CARNICOS_KNOWLEDGE_PATH")
    if env_path:
        return absolute_project_path(Path(env_path))

    chunks_path = PROJECT_ROOT / DEFAULT_CHUNKS_FILE
    if chunks_path.exists():
        return chunks_path

    return PROJECT_ROOT / DEFAULT_DATASET_DIR


def absolute_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_local_checks(knowledge_path: Path) -> bool:
    checks_ok = True

    dataset_dir = PROJECT_ROOT / DEFAULT_DATASET_DIR
    if dataset_dir.exists():
        markdown_count = len(sorted(dataset_dir.glob("*.md")))
        print_ok(f"Corpus Markdown encontrado: {relative_path(dataset_dir)} ({markdown_count} archivos .md)")
        checks_ok = checks_ok and markdown_count > 0
        if markdown_count == 0:
            print_error("El corpus existe, pero no contiene archivos .md.")
    else:
        print_error(f"No existe el corpus procesado: {relative_path(dataset_dir)}")
        checks_ok = False

    if not knowledge_path.exists():
        print_error(f"No existe la base de conocimiento seleccionada: {relative_path(knowledge_path)}")
        return False

    if knowledge_path.is_file():
        chunk_count = count_chunk_headings(knowledge_path)
        size_bytes = knowledge_path.stat().st_size
        print_ok(
            "Archivo consolidado encontrado: "
            f"{relative_path(knowledge_path)} ({size_bytes:,} bytes, {chunk_count} chunks)"
        )
        checks_ok = checks_ok and chunk_count > 0
        if chunk_count == 0:
            print_error("El archivo existe, pero no se detectaron encabezados de chunks.")
    else:
        markdown_count = len(sorted(knowledge_path.glob("*.md")))
        print_ok(
            "Directorio de conocimiento seleccionado: "
            f"{relative_path(knowledge_path)} ({markdown_count} archivos .md)"
        )
        checks_ok = checks_ok and markdown_count > 0
        if markdown_count == 0:
            print_error("El directorio seleccionado no contiene archivos .md.")

    try:
        knowledge_base = load_knowledge_base(str(knowledge_path), verbose=False)
    except Exception as exc:
        print_error(f"No fue posible cargar la base de conocimiento: {exc}")
        return False

    stats = get_knowledge_stats(knowledge_base)
    print_ok(
        "Carga de conocimiento exitosa: "
        f"{stats['total_characters']:,} caracteres, "
        f"{stats['total_words']:,} palabras, "
        f"{stats['total_paragraphs']:,} parrafos"
    )
    return checks_ok


def run_llm_check(knowledge_path: Path, args: argparse.Namespace) -> bool | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    max_tokens = env_int("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    temperature = env_float("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE)

    print()
    print("Prueba contra el LLM")
    print("-" * 72)
    print(f"Modelo configurado: {model}")
    print(f"Temperatura: {temperature}")
    print(f"Max tokens: {max_tokens}")

    if args.skip_llm:
        print_skip("Prueba LLM omitida por parametro --skip-llm.")
        return None

    if not looks_like_real_api_key(api_key):
        message = (
            "OPENAI_API_KEY no esta configurada con una clave real. "
            "La validacion local queda aprobada, pero falta la prueba end-to-end."
        )
        if args.require_llm:
            print_error(message)
            return False
        print_skip(message)
        return None

    try:
        qa_system = CarnicosQASystem(
            knowledge_dir=str(knowledge_path),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=False,
        )
        answer = qa_system.answer(args.question)
    except Exception as exc:
        print_error(f"No fue posible inicializar o consultar el sistema Q&A: {exc}")
        return False

    if not answer.strip() or answer.startswith("Error al procesar la pregunta"):
        print_error(f"La prueba LLM no produjo una respuesta valida: {answer}")
        return False

    print_ok("Respuesta generada correctamente por el sistema Q&A.")
    print("Pregunta de prueba:")
    print(f"  {args.question}")
    print("Respuesta resumida:")
    print(f"  {compact_answer(answer)}")
    return True


def count_chunk_headings(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    return len(CHUNK_HEADING_RE.findall(content))


def looks_like_real_api_key(api_key: str) -> bool:
    return api_key.startswith("sk-") and len(api_key) > 20 and not api_key.startswith("sk_test")


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def compact_answer(answer: str, max_length: int = 500) -> str:
    compact = " ".join(answer.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def print_ok(message: str) -> None:
    print(f"[OK] {message}")


def print_skip(message: str) -> None:
    print(f"[OMITIDO] {message}")


def print_error(message: str) -> None:
    print(f"[ERROR] {message}")


def print_summary(result: ValidationResult) -> None:
    print()
    print("Resultado")
    print("-" * 72)

    if result.exit_code == 0 and result.llm_check_ok is True:
        print("[OK] Validacion completa aprobada: base local y llamada al LLM funcionando.")
    elif result.exit_code == 0:
        print(
            "[OK] Validacion local aprobada. Para cierre end-to-end, configura "
            "OPENAI_API_KEY y ejecuta nuevamente con --require-llm."
        )
    else:
        print("[ERROR] La validacion no esta completa. Revisa los mensajes anteriores.")


if __name__ == "__main__":
    main()
