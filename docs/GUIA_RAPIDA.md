# Guia Rapida — Asistente Conversacional Alimentos Carnicos S.A.S.

Estado actual: agente LangGraph ReAct con RAG vectorial, API FastAPI, canal WhatsApp via Twilio + N8N + ngrok y escalada automatica a asesor humano.

---

## Inicio rapido

### 1. Configurar variables de entorno

Copia `.env.example` a `.env` y completa los valores obligatorios:

```bash
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=openai:gpt-4o-mini
DATABASE_URL=postgresql://user:pass@localhost:5432/carnicos_kb   # opcional
HITL_ENABLED=false
```

### 2. Instalar dependencias

```powershell
make sync
```

### 3. Construir la base de conocimiento (primera vez)

```powershell
make scrape       # extrae paginas del sitio web
make pdf-fast     # convierte PDFs a Markdown
make chunk        # genera los chunks semanticos
make rag-index    # construye el indice vectorial
```

### 4. Levantar el sistema

```powershell
# Terminal 1 — API del agente
make api

# Terminal 2 — tunel HTTPS publico
make ngrok

# Opcional — interfaz Streamlit
make app
```

---

## Endpoints de la API

| Metodo | Ruta | Descripcion |
|---|---|---|
| POST | `/chat` | Enviar pregunta al agente |
| POST | `/chat/resume` | Reanudar flujo tras decision HITL |
| GET | `/health` | Estado del servicio |

Ejemplo rapido con PowerShell:

```powershell
$url = (Invoke-RestMethod http://localhost:4040/api/tunnels).tunnels[0].public_url
$body = @{ message = "Que productos ofrece Rica?"; phone_number = "573001234567" } | ConvertTo-Json
Invoke-RestMethod -Uri "$url/chat" -Method Post -Body $body -ContentType "application/json" -Headers @{"ngrok-skip-browser-warning"="1"}
```

Respuesta esperada:

```json
{
  "answer": "...",
  "confidence": "high",
  "tool_was_called": true,
  "pending_approval": false,
  "thread_id": "573001234567"
}
```

---

## Flujo N8N — como probar

El flujo tiene 6 nodos:

```
Webhook -> Edit Fields -> HTTP Request -> IF -> Mensaje al asesor -> Mensaje al cliente
                                            -> Respuesta del agente
```

El nodo IF evalua `confidence == "low"`:
- **FALSE** (confidence alta): el cliente recibe la respuesta del agente directamente.
- **TRUE** (confidence baja): el asesor recibe alerta con contexto y el cliente recibe confirmacion de atencion.

Para probar la rama TRUE envia desde WhatsApp:

```
¿Cual es el precio actual del kilo de salchicha Rica?
```

Para probar la rama FALSE envia:

```
¿Que marcas tiene Alimentos Carnicos?
```

---

## Control humano HITL (opcional)

Activar en `.env`:

```bash
HITL_ENABLED=true
```

Con HITL activo, el agente pausa antes de consultar el RAG y espera que el operador apruebe, edite o rechace la query desde la interfaz Streamlit. Ver `docs/hitl_middleware.md` para detalle completo.

---

## Referencia rapida de archivos

| Archivo | Funcion |
|---|---|
| `src/carnicos_kb/qa_system.py` | Agente LangGraph, HITL, checkpointer |
| `src/carnicos_kb/api.py` | Endpoints FastAPI |
| `src/carnicos_kb/vector_retriever_tool.py` | Herramienta RAG con PGVector / InMemory |
| `src/carnicos_kb/streamlit_app.py` | Interfaz web con panel HITL |
| `docs/informe_tecnico_final.md` | Documentacion consolidada del proyecto |
| `docs/hitl_middleware.md` | Detalle tecnico del middleware HITL |
| `docs/definicion_alcance.md` | Alcance y preguntas de validacion para N8N |
