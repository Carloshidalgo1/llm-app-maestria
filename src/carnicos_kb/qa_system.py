"""
Sistema de Preguntas y Respuestas (Q&A) usando LangChain y OpenAI GPT-5.4-nano.

Este módulo implementa un sistema Q&A robusto basado en la base de conocimiento
consolidada. Utiliza prompt engineering avanzado para evitar alucinaciones y
asegurar respuestas precisas.
"""

import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .knowledge_loader import load_knowledge_base, get_knowledge_stats

# Cargar variables de entorno
load_dotenv()


class CarnicosQASystem:
    """
    Sistema Q&A para Alimentos Cárnicos usando LangChain y OpenAI.
    """
    
    def __init__(
        self,
        knowledge_dir: str = None,
        model: str = "gpt-5.4-nano",
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ):
        """
        Inicializa el sistema Q&A.
        
        Args:
            knowledge_dir: Directorio con la base de conocimiento (busca automáticamente si es None)
            model: Modelo de OpenAI a utilizar
            temperature: Temperatura del modelo (0-1, menor = más determinístico)
            max_tokens: Máximo número de tokens en la respuesta
        """
        # Si no se especifica directorio, buscar el archivo de chunks o el directorio de dataset
        if knowledge_dir is None:
            posibles_ubicaciones = [
                Path("data/processed/base_conocimiento_chunks.md"),
                #Path("base_conocimiento_chunks.md"),
                #Path("data/processed/dataset_carnicos"),
                #Path("dataset_carnicos"),
                #Path("../../data/processed/dataset_carnicos"),  # Desde src/carnicos_kb/
            ]

            knowledge_dir = None
            for ubicacion in posibles_ubicaciones:
                if ubicacion.exists():
                    knowledge_dir = str(ubicacion)
                    break

            if knowledge_dir is None:
                raise FileNotFoundError(
                    "No se encontró la base de conocimiento. "
                    "Buscando: data/processed/base_conocimiento_chunks.md, base_conocimiento_chunks.md, dataset_carnicos, data/processed/dataset_carnicos"
                )
        
        self.knowledge_dir = knowledge_dir
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Validar API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no está configurada. "
                "Por favor, agrega la clave en el archivo .env"
            )
        
        # Inicializar LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        
        # Cargar base de conocimiento
        print(f"\n🔄 Inicializando sistema Q&A...")
        print(f"📂 Cargando base de conocimiento desde: {knowledge_dir}")
        self.knowledge_base = load_knowledge_base(knowledge_dir)
        
        # Mostrar estadísticas
        stats = get_knowledge_stats(self.knowledge_base)
        print(f"\n📊 Estadísticas de la base de conocimiento:")
        print(f"  • Caracteres totales: {stats['total_characters']:,}")
        print(f"  • Palabras totales: {stats['total_words']:,}")
        print(f"  • Párrafos totales: {stats['total_paragraphs']:,}")
        print(f"\n✅ Sistema Q&A inicializado correctamente")
    
    def _create_system_prompt(self) -> str:
        """
        Crea un prompt de sistema robusto que consolida la base de conocimiento
        e instruye al modelo sobre cómo responder.
        
        Returns:
            Prompt de sistema consolidado
        """
        system_prompt = f"""Eres un asistente experto y muy preciso sobre la empresa Alimentos Cárnicos.

TU BASE DE CONOCIMIENTO (información verificada sobre la empresa):
========================================================================
{self.knowledge_base}
========================================================================

INSTRUCCIONES CRÍTICAS PARA TUS RESPUESTAS:

1. RESPONDE SOLO CON INFORMACIÓN VERIFICADA: 
   - Utiliza ÚNICAMENTE la información de la base de conocimiento anterior.
   - Si la pregunta requiere información que NO está en la base de conocimiento, 
     responde: "No tengo información verificada sobre esto en mi base de conocimiento."

2. EVITA ALUCINACIONES:
   - NO inventes datos, números, fechas o hechos que no estén explícitamente mencionados.
   - NO hagas suposiciones ni extrapolaciones.
   - Si no estás seguro, admítelo claramente.

3. CLARIDAD Y PRECISIÓN:
   - Sé conciso pero completo en tus respuestas.
   - Si hay múltiples respuestas, preséntalas de forma ordenada.
   - Siempre cita la fuente de tu información cuando sea posible.

4. TONO Y PROFESIONALISMO:
   - Mantén un tono profesional, amable y servicial.
   - Adapta tu respuesta al contexto de la pregunta.
   - Si es una pregunta de contacto, proporciona la información de manera clara.

5. FORMATO:
   - Usa listas con viñetas (•) cuando sea apropiado.
   - Resalta información clave en negrilla (**texto**).
   - Estructura las respuestas para fácil lectura.

RECUERDA: Tu rol es ser un canal de comunicación preciso y confiable para Alimentos Cárnicos.
La calidad y precisión de tus respuestas son más importantes que la extensión."""
        
        return system_prompt
    
    def answer(self, question: str) -> str:
        """
        Responde una pregunta usando el LLM con el contexto de la base de conocimiento.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Respuesta generada por el modelo
        """
        if not question.strip():
            return "Por favor, formula una pregunta válida."
        
        system_prompt = self._create_system_prompt()
        
        # Crear mensajes para el modelo
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        
        try:
            # Invocar el modelo
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"❌ Error al procesar la pregunta: {str(e)}"
    
    def interactive_chat(self):
        """
        Inicia un chat interactivo donde el usuario puede hacer preguntas.
        """
        print("\n" + "="*70)
        print("🤖 ASISTENTE Q&A - ALIMENTOS CÁRNICOS")
        print("="*70)
        print("Escribe 'salir' para terminar la conversación.")
        print("-"*70 + "\n")
        
        while True:
            question = input("❓ Tu pregunta: ").strip()
            
            if question.lower() in ["salir", "exit", "quit"]:
                print("\n👋 ¡Gracias por usar el asistente! Hasta luego.")
                break
            
            if not question:
                print("⚠️ Por favor, ingresa una pregunta válida.\n")
                continue
            
            print("\n⏳ Procesando pregunta...\n")
            answer = self.answer(question)
            print(f"🤖 Respuesta:\n{answer}\n")
            print("-"*70 + "\n")


def main():
    """
    Función principal para ejecutar el sistema Q&A.
    """
    import sys
    
    try:
        qa_system = CarnicosQASystem()
        
        # Si hay preguntas como argumentos, responderlas
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
            print(f"\n❓ Pregunta: {question}\n")
            answer = qa_system.answer(question)
            print(f"🤖 Respuesta:\n{answer}\n")
        else:
            # Iniciar chat interactivo
            qa_system.interactive_chat()
            
    except ValueError as e:
        print(f"❌ Error de configuración: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
