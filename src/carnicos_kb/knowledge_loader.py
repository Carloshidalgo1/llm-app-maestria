"""
Módulo para cargar y consolidar la base de conocimiento desde archivos .md
generados por el web scraping.
"""

import os
from pathlib import Path
from typing import Dict, List


def load_knowledge_base(knowledge_dir: str = "dataset_carnicos") -> str:
    """
    Carga todos los archivos .md del directorio de conocimiento
    y los consolida en un único string.
    
    Args:
        knowledge_dir: Ruta al directorio con los archivos .md
        
    Returns:
        String consolidado con todo el contenido del conocimiento
    """
    knowledge_content = []
    knowledge_path = Path(knowledge_dir)
    
    if not knowledge_path.exists():
        raise FileNotFoundError(f"Ruta de conocimiento no encontrada: {knowledge_dir}")
    
    if knowledge_path.is_file():
        # Cargar un único archivo .md de chunks
        if knowledge_path.suffix.lower() != ".md":
            raise FileNotFoundError(f"El archivo de conocimiento debe ser .md: {knowledge_dir}")
        print(f"Cargando archivo de conocimiento: {knowledge_path.name}")
        with open(knowledge_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise FileNotFoundError(f"El archivo de conocimiento está vacío: {knowledge_dir}")
            knowledge_content.append(f"--- Fuente: {knowledge_path.name} ---\n{content}")
    else:
        # Obtener todos los archivos .md en el directorio
        md_files = sorted(knowledge_path.glob("*.md"))
        
        if not md_files:
            raise FileNotFoundError(f"No hay archivos .md en {knowledge_dir}")
        
        print(f"📚 Cargando {len(md_files)} archivos de conocimiento...")
        
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        knowledge_content.append(f"--- Fuente: {md_file.name} ---\n{content}")
            except Exception as e:
                print(f"⚠️ Error al leer {md_file.name}: {e}")
                continue
    
    consolidated = "\n\n".join(knowledge_content)
    print(f"Base de conocimiento consolidada: {len(consolidated)} caracteres")
    
    return consolidated


def get_knowledge_stats(knowledge_base: str) -> Dict[str, any]:
    """
    Retorna estadísticas sobre la base de conocimiento.
    
    Args:
        knowledge_base: String con el contenido consolidado
        
    Returns:
        Diccionario con estadísticas
    """
    return {
        "total_characters": len(knowledge_base),
        "total_words": len(knowledge_base.split()),
        "total_paragraphs": knowledge_base.count("\n\n"),
    }


if __name__ == "__main__":
    kb = load_knowledge_base()
    stats = get_knowledge_stats(kb)
    print(f"\n📊 Estadísticas de la base de conocimiento:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
