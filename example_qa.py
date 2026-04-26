"""
Ejemplo de uso del Sistema Q&A de Alimentos Cárnicos.

Este script demuestra cómo usar el sistema Q&A de forma programática.
"""

from src.carnicos_kb.qa_system import CarnicosQASystem


def main():
    """
    Función principal que demuestra el uso del sistema Q&A.
    """
    
    print("="*70)
    print("📚 EJEMPLO: Sistema Q&A para Alimentos Cárnicos")
    print("="*70)
    
    # Inicializar el sistema
    print("\n🔄 Inicializando sistema Q&A...\n")
    qa = CarnicosQASystem()
    
    # Preguntas de prueba
    preguntas_prueba = [
        "¿Cuál es la misión de Alimentos Cárnicos?",
        "¿Qué productos ofrecen?",
        "¿Cuál es la información de contacto?",
        "¿Dónde está ubicada la empresa?",
        "¿Cuál es el compromiso con la sostenibilidad?",
    ]
    
    print("\n" + "="*70)
    print("🧪 PRUEBAS DE PREGUNTAS")
    print("="*70)
    
    for i, pregunta in enumerate(preguntas_prueba, 1):
        print(f"\n📝 Pregunta {i}: {pregunta}")
        print("-" * 70)
        
        respuesta = qa.answer(pregunta)
        print(f"🤖 Respuesta:\n{respuesta}")
        print()
    
    print("="*70)
    print("✅ Pruebas completadas")
    print("="*70)
    
    # Opción de chat interactivo
    print("\n¿Deseas iniciar chat interactivo? (escribe 'si' o 'no'): ", end="")
    respuesta_usuario = input().strip().lower()
    
    if respuesta_usuario in ["si", "yes", "y", "sí"]:
        qa.interactive_chat()


if __name__ == "__main__":
    main()
