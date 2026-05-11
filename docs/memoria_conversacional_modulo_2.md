# Memoria conversacional - Modulo 2

## Objetivo

La memoria conversacional permite que el asistente responda preguntas de
seguimiento dentro de una misma sesion. Antes de este cambio, Streamlit mostraba
el historial visualmente, pero `CarnicosQASystem.answer()` solo recibia la
pregunta actual. Ahora el historial previo tambien se envia al LLM como parte
del contexto.

## Implementacion

La memoria se implemento en dos niveles:

- `src/carnicos_kb/qa_system.py`: normaliza el historial a mensajes de
  LangChain (`HumanMessage` y `AIMessage`) y construye el arreglo enviado al LLM
  con este orden: prompt de sistema, historial y pregunta actual.
- `src/carnicos_kb/streamlit_app.py`: conserva los mensajes de la sesion en
  `st.session_state.messages` y los pasa a `qa_system.answer()` antes de
  generar cada nueva respuesta.

El sistema usa `InMemoryChatMessageHistory` de `langchain_core` para soportar
memoria interna en el chat por consola. En Streamlit se usa el historial de
sesion para evitar mezclar conversaciones entre usuarios o recargas.

## Beneficios

- Permite preguntas como "cual fue el primero que mencionaste?".
- Mantiene coherencia en conversaciones de varios turnos.
- Conserva las reglas del prompt original: la base de conocimiento sigue siendo
  la fuente prioritaria frente al historial.

## Limitaciones

- La memoria vive solo durante la sesion actual.
- No persiste en base de datos.
- Si la conversacion crece demasiado, el prompt puede aumentar de tamano.
- La memoria no reemplaza la base de conocimiento: solo ayuda a interpretar
  referencias y continuidad del dialogo.

## Prueba sugerida para sustentacion

1. Preguntar: `Que compromisos existen sobre bienestar animal?`
2. Luego preguntar: `Cual fue el primero que mencionaste?`
3. Verificar que la segunda respuesta use la respuesta anterior como contexto y
   no trate la pregunta como aislada.

## Validacion realizada

Se ejecuto una prueba real de dos turnos con `CarnicosQASystem`:

```powershell
.\.venv\Scripts\python.exe -c "from carnicos_kb.qa_system import CarnicosQASystem; qa=CarnicosQASystem(verbose=False); a1=qa.answer('Que compromisos existen sobre bienestar animal?'); a2=qa.answer('Cual fue el primero que mencionaste?'); print(a2); print(len(qa.get_memory_messages()))"
```

Resultado observado:

- La primera respuesta enumero compromisos de bienestar animal.
- La segunda respuesta identifico que "el primero" era la meta 2027 sobre
  cerdas en gestacion libres de jaulas.
- La memoria interna conservo 4 mensajes: usuario, asistente, usuario y
  asistente.
