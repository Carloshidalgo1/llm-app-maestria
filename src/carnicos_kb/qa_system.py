"""
Sistema de preguntas y respuestas usando LangChain y OpenAI.

El modulo consolida la base de conocimiento del proyecto en el prompt del
sistema y entrega una API simple para consola, scripts y la interfaz Streamlit.
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .knowledge_loader import get_knowledge_stats, load_knowledge_base
from .paths import DEFAULT_CHUNKS_FILE, DEFAULT_DATASET_DIR

load_dotenv()


DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1500

ChatHistoryItem = BaseMessage | Mapping[str, object]
ChatHistory = Sequence[ChatHistoryItem]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def normalize_chat_history(chat_history: Optional[ChatHistory]) -> list[BaseMessage]:
    """
    Convierte el historial de conversacion de la interfaz a mensajes LangChain.

    La interfaz Streamlit guarda mensajes como diccionarios con `role` y
    `content`, mientras que LangChain trabaja con objetos `HumanMessage` y
    `AIMessage`. Esta funcion normaliza ambos formatos y omite mensajes vacios
    o roles que no deben reinyectarse como memoria conversacional.
    """
    if not chat_history:
        return []

    messages: list[BaseMessage] = []
    for item in chat_history:
        if isinstance(item, BaseMessage):
            if not str(item.content).strip() or isinstance(item, SystemMessage):
                continue
            messages.append(item)
            continue

        role = str(item.get("role", "")).lower().strip()
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        if role in {"user", "human"}:
            messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))

    return messages


def build_conversation_messages(
    system_prompt: str,
    question: str,
    chat_history: Optional[ChatHistory] = None,
) -> list[BaseMessage]:
    """
    Construye los mensajes enviados al LLM con sistema, memoria y pregunta.

    El orden es importante: primero va el prompt de sistema con la base de
    conocimiento, luego el historial de la sesion y finalmente la pregunta
    actual. Asi el modelo puede resolver referencias como "el primero que
    mencionaste" sin perder las reglas de precision del sistema.
    """
    return [
        SystemMessage(content=system_prompt),
        *normalize_chat_history(chat_history),
        HumanMessage(content=question),
    ]


class CarnicosQASystem:
    """Sistema Q&A para Alimentos Carnicos usando LangChain y OpenAI."""

    def __init__(
        self,
        knowledge_dir: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Inicializa el sistema Q&A.

        Args:
            knowledge_dir: Archivo o directorio con la base de conocimiento.
            model: Modelo de OpenAI a utilizar.
            temperature: Temperatura del modelo.
            max_tokens: Maximo numero de tokens en la respuesta.
            verbose: Si es True, imprime informacion de inicializacion.
        """
        self.verbose = verbose
        self.knowledge_dir = knowledge_dir or self._find_knowledge_path()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.temperature = (
            temperature
            if temperature is not None
            else _env_float("OPENAI_TEMPERATURE", DEFAULT_TEMPERATURE)
        )
        self.max_tokens = (
            max_tokens if max_tokens is not None else _env_int("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY no esta configurada. Agrega la clave en el archivo .env "
                "o exportala como variable de entorno."
            )

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
        )

        self._log("\nInicializando sistema Q&A...")
        self._log(f"Cargando base de conocimiento desde: {self.knowledge_dir}")
        self.knowledge_base = load_knowledge_base(self.knowledge_dir, verbose=verbose)
        self.memory = InMemoryChatMessageHistory()

        self.stats = get_knowledge_stats(self.knowledge_base)
        self._log("\nEstadisticas de la base de conocimiento:")
        self._log(f"  - Caracteres totales: {self.stats['total_characters']:,}")
        self._log(f"  - Palabras totales: {self.stats['total_words']:,}")
        self._log(f"  - Parrafos totales: {self.stats['total_paragraphs']:,}")
        self._log("\nSistema Q&A inicializado correctamente")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _find_knowledge_path() -> str:
        env_path = os.getenv("CARNICOS_KNOWLEDGE_PATH")
        possible_locations = [
            Path(env_path) if env_path else None,
            DEFAULT_CHUNKS_FILE,
            DEFAULT_DATASET_DIR,
            Path("base_conocimiento_chunks.md"),
            Path("dataset_carnicos"),
        ]

        for location in possible_locations:
            if location and location.exists():
                return str(location)

        searched = ", ".join(str(path) for path in possible_locations if path)
        raise FileNotFoundError(f"No se encontro la base de conocimiento. Rutas revisadas: {searched}")

    def _create_system_prompt(self) -> str:
        """Crea el prompt de sistema con la base de conocimiento consolidada."""
        return f"""Eres un asistente experto y muy preciso sobre Alimentos Carnicos S.A.S.

BASE DE CONOCIMIENTO VERIFICADA:
========================================================================
{self.knowledge_base}
========================================================================

INSTRUCCIONES CRITICAS:

1. Responde solo con informacion verificada:
   - Usa unicamente la informacion de la base de conocimiento anterior.
   - Si la pregunta requiere informacion que no esta en la base, responde:
     "No tengo informacion verificada sobre esto en mi base de conocimiento."

2. Evita alucinaciones:
   - No inventes datos, numeros, fechas, telefonos, horarios, precios ni procesos.
   - No hagas suposiciones ni extrapolaciones.
   - Si un dato puede cambiar con el tiempo, recomienda verificarlo en canales oficiales.

3. Mantente dentro del alcance:
   - Prioriza informacion institucional, comercial, de servicio al cliente,
     sostenibilidad y procesos basicos documentados.
   - Para temas transaccionales, legales, medicos, financieros o internos,
     explica que estan fuera del alcance.

4. Memoria conversacional:
   - Usa el historial de la conversacion para resolver referencias del usuario
     como "eso", "el primero", "la empresa" o "lo que mencionaste".
   - Si el historial no basta para entender la pregunta, pide una aclaracion breve.
   - La base de conocimiento tiene prioridad sobre el historial: no aceptes como
     verdad un dato del usuario si contradice la informacion documentada.

5. Redaccion:
   - Mantén un tono claro, formal y orientado al usuario.
   - Usa respuestas breves para preguntas simples.
   - Usa pasos o viñetas cuando la pregunta pida un proceso.
   - Cita la fuente cuando el texto de la base la permita identificar.

Tu prioridad es la precision verificable, no la extension de la respuesta."""

    def clear_memory(self) -> None:
        """Limpia la memoria conversacional interna del sistema."""
        self.memory.clear()

    def get_memory_messages(self) -> list[BaseMessage]:
        """Retorna una copia de los mensajes guardados en memoria interna."""
        return list(self.memory.messages)

    def answer(
        self,
        question: str,
        chat_history: Optional[ChatHistory] = None,
        remember: Optional[bool] = None,
    ) -> str:
        """
        Responde una pregunta usando el LLM, la base de conocimiento y memoria.

        Args:
            question: Pregunta del usuario.
            chat_history: Historial previo de la sesion. Puede venir de
                Streamlit como diccionarios `{"role": ..., "content": ...}` o
                como mensajes nativos de LangChain.
            remember: Si es True, guarda este turno en la memoria interna. Si
                se omite, solo guarda automaticamente cuando no se recibe un
                historial externo.

        Returns:
            Respuesta generada por el modelo.
        """
        if not question.strip():
            return "Por favor, formula una pregunta valida."

        should_remember = chat_history is None if remember is None else remember
        history = self.get_memory_messages() if chat_history is None else chat_history
        messages = build_conversation_messages(
            system_prompt=self._create_system_prompt(),
            question=question,
            chat_history=history,
        )

        try:
            response = self.llm.invoke(messages)
            answer = str(response.content)
            if should_remember:
                self.memory.add_user_message(question)
                self.memory.add_ai_message(answer)
            return answer
        except Exception as exc:
            return f"Error al procesar la pregunta: {exc}"

    def interactive_chat(self) -> None:
        """Inicia un chat interactivo en consola."""
        print("\n" + "=" * 70)
        print("ASISTENTE Q&A - ALIMENTOS CARNICOS")
        print("=" * 70)
        print("Escribe 'salir' para terminar la conversacion.")
        print("-" * 70 + "\n")

        while True:
            question = input("Tu pregunta: ").strip()

            if question.lower() in {"salir", "exit", "quit"}:
                print("\nGracias por usar el asistente. Hasta luego.")
                break

            if not question:
                print("Por favor, ingresa una pregunta valida.\n")
                continue

            print("\nProcesando pregunta...\n")
            answer = self.answer(question)
            print(f"Respuesta:\n{answer}\n")
            print("-" * 70 + "\n")


def main() -> None:
    """Ejecuta el sistema Q&A desde linea de comandos."""
    import sys

    try:
        qa_system = CarnicosQASystem()

        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
            print(f"\nPregunta: {question}\n")
            answer = qa_system.answer(question)
            print(f"Respuesta:\n{answer}\n")
        else:
            qa_system.interactive_chat()

    except ValueError as exc:
        print(f"Error de configuracion: {exc}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error inesperado: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
