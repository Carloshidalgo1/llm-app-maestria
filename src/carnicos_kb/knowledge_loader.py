"""
Modulo para cargar y consolidar la base de conocimiento desde archivos .md.
"""

from pathlib import Path
from typing import Dict


def load_knowledge_base(knowledge_dir: str = "dataset_carnicos", verbose: bool = True) -> str:
    """
    Carga todos los archivos .md del directorio de conocimiento
    y los consolida en un unico string.

    Args:
        knowledge_dir: Ruta al directorio con los archivos .md.
        verbose: Si es True, imprime informacion de carga.

    Returns:
        String consolidado con todo el contenido del conocimiento.
    """
    knowledge_content = []
    knowledge_path = Path(knowledge_dir)

    if not knowledge_path.exists():
        raise FileNotFoundError(f"Ruta de conocimiento no encontrada: {knowledge_dir}")

    if knowledge_path.is_file():
        if knowledge_path.suffix.lower() != ".md":
            raise FileNotFoundError(f"El archivo de conocimiento debe ser .md: {knowledge_dir}")

        if verbose:
            print(f"Cargando archivo de conocimiento: {knowledge_path.name}")

        content = knowledge_path.read_text(encoding="utf-8").strip()
        if not content:
            raise FileNotFoundError(f"El archivo de conocimiento esta vacio: {knowledge_dir}")
        knowledge_content.append(f"--- Fuente: {knowledge_path.name} ---\n{content}")
    else:
        md_files = sorted(knowledge_path.glob("*.md"))

        if not md_files:
            raise FileNotFoundError(f"No hay archivos .md en {knowledge_dir}")

        if verbose:
            print(f"Cargando {len(md_files)} archivos de conocimiento...")

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8").strip()
            except Exception as exc:
                if verbose:
                    print(f"Error al leer {md_file.name}: {exc}")
                continue

            if content:
                knowledge_content.append(f"--- Fuente: {md_file.name} ---\n{content}")

    consolidated = "\n\n".join(knowledge_content)
    if verbose:
        print(f"Base de conocimiento consolidada: {len(consolidated)} caracteres")

    return consolidated


def get_knowledge_stats(knowledge_base: str) -> Dict[str, int]:
    """
    Retorna estadisticas sobre la base de conocimiento.

    Args:
        knowledge_base: String con el contenido consolidado.

    Returns:
        Diccionario con estadisticas.
    """
    return {
        "total_characters": len(knowledge_base),
        "total_words": len(knowledge_base.split()),
        "total_paragraphs": knowledge_base.count("\n\n"),
    }


if __name__ == "__main__":
    kb = load_knowledge_base()
    stats = get_knowledge_stats(kb)
    print("\nEstadisticas de la base de conocimiento:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
