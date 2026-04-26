# Sistema Q&A - Punto 3: Construcción del Aplicativo

## Descripción

Este módulo implementa un **Sistema de Preguntas y Respuestas (Q&A)** para Alimentos Cárnicos utilizando:

- **Framework**: LangChain (orquestación de LLM)
- **Modelo**: OpenAI GPT (gpt-4o-mini por defecto, configurable)
- **Técnicas**: Prompt Engineering avanzado, zero-shot reasoning, evitar alucinaciones
- **Base de Conocimiento**: Todo el contenido limpio del punto 2 (web scraping)

## Características principales

✅ **Base de Conocimiento Consolidada**: Carga automática de todos los archivos .md del corpus  
✅ **Prompt Robusto**: Sistema de prompts diseñado para evitar alucinaciones  
✅ **Chat Interactivo**: Interface para hacer preguntas en tiempo real  
✅ **API Flexible**: Uso programático desde Python  
✅ **Estadísticas**: Información sobre la base de conocimiento cargada  

## Configuración Inicial

### 1. Obtener API Key de OpenAI

1. Ve a [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Crea una nueva API key
3. Copia la clave (solo la verás una vez)

### 2. Configurar .env

Edita el archivo `.env` y reemplaza:

```bash
OPENAI_API_KEY=sk_test_tu_api_key_aqui
```

Por tu clave real:

```bash
OPENAI_API_KEY=sk-proj-tu_clave_aqui
```

### 3. Instalar dependencias

```bash
# Con uv
make sync

# O con pip
make install-pip
```

## Uso

### Opción 1: Chat Interactivo

Inicia el asistente interactivo:

```bash
make qa
```

Luego escribe tus preguntas:

```
❓ Tu pregunta: ¿Cuál es la misión de Alimentos Cárnicos?
⏳ Procesando pregunta...
🤖 Respuesta: [respuesta del modelo]
```

Escribe `salir` para terminar.

### Opción 2: Pregunta Única

Haz una pregunta directa:

```bash
uv run carnicos-qa "¿Qué productos ofrece Alimentos Cárnicos?"
```

O en Windows:

```bash
uv run carnicos-qa ¿Qué productos ofrece Alimentos Cárnicos?
```

### Opción 3: Uso Programático

```python
from src.carnicos_kb.qa_system import CarnicosQASystem

# Inicializar el sistema
qa = CarnicosQASystem()

# Hacer una pregunta
respuesta = qa.answer("¿Cuál es la información de contacto?")
print(respuesta)

# Chat interactivo
qa.interactive_chat()
```

## Configuración del Modelo

Edita el archivo `.env` para ajustar parámetros:

```bash
# Modelo (cambiar si tienes acceso a otros modelos)
OPENAI_MODEL=gpt-4o-mini

# Temperatura (0=determinístico, 1=creativo)
# Recomendado: 0.2 para Q&A (respuestas precisas)
OPENAI_TEMPERATURE=0.2

# Máximo tokens en respuesta
OPENAI_MAX_TOKENS=1500
```

## Estructura del Sistema

### Knowledge Loader (`knowledge_loader.py`)

```python
from src.carnicos_kb.knowledge_loader import load_knowledge_base

# Cargar base de conocimiento
kb = load_knowledge_base("dataset_carnicos")
```

- Carga todos los archivos .md del directorio
- Consolida el contenido en un único string
- Proporciona estadísticas

### Q&A System (`qa_system.py`)

```python
from src.carnicos_kb.qa_system import CarnicosQASystem

qa = CarnicosQASystem()
respuesta = qa.answer("Tu pregunta aquí")
```

- `__init__()`: Inicializa con la base de conocimiento
- `answer()`: Responde una pregunta usando el LLM
- `interactive_chat()`: Inicia chat interactivo

## Estrategia de Prompt Engineering

El sistema implementa varias técnicas para evitar alucinaciones:

### 1. **Contexto Consolidado**
Toda la base de conocimiento se incluye en el prompt de sistema, asegurando que el modelo tenga información verificada.

### 2. **Instrucciones Explícitas**
- "Responde SOLO con información verificada"
- "Si no está en la base de conocimiento, admítelo"
- "NO inventes datos"

### 3. **Zero-shot Reasoning**
El modelo responde basándose únicamente en el contexto proporcionado, sin aprender nuevas tareas.

### 4. **Temperatura Baja**
Temperature = 0.2 hace que el modelo sea más determinístico y preciso.

### 5. **Auditoría de Respuestas**
Se recomienda verificar que las respuestas citen fuentes de la base de conocimiento.

## Pruebas Recomendadas

Según los requisitos, realiza al menos 20 preguntas para validar:

### Categoría: Información Institucional
1. ¿Cuál es la misión de Alimentos Cárnicos?
2. ¿Cuál es la visión de la empresa?
3. ¿Cuándo fue fundada la empresa?
4. ¿Dónde está ubicada la empresa?

### Categoría: Productos y Servicios
5. ¿Qué tipos de productos ofrece?
6. ¿Cuáles son las características de los productos?
7. ¿Dónde puedo comprar?

### Categoría: Sostenibilidad
8. ¿Cuál es el compromiso de la empresa con el medio ambiente?
9. ¿Tiene certificaciones de sostenibilidad?

### Categoría: Contacto
10. ¿Cuál es el número de teléfono?
11. ¿Cuál es el correo de contacto?
12. ¿Cuál es la dirección?

## Estructura de Archivos

```
src/carnicos_kb/
├── qa_system.py           # Sistema Q&A principal
├── knowledge_loader.py    # Cargador de base de conocimiento
├── scraper.py             # Web scraper
└── ...

dataset_carnicos/          # Base de conocimiento (corpus)
├── alimentoscarnicos.com.co.md
├── nosotros.md
├── mision-vision.md
└── ... (36 archivos)

.env                       # Variables de entorno (OPENAI_API_KEY, etc.)
```

## Solución de Problemas

### Error: "OPENAI_API_KEY no está configurada"
- Verifica que el `.env` tiene la clave correcta
- Reinicia la terminal después de editar `.env`

### Error: "No hay archivos .md en dataset_carnicos"
- Ejecuta `make scrape` primero para generar el corpus
- Verifica que los archivos están en `dataset_carnicos/`

### Respuestas vagas o incorrectas
- Aumenta la temperatura a 0.3-0.5 para más creatividad
- Revisa que la base de conocimiento tiene la información requerida
- Reformula la pregunta con más contexto

## Próximos Pasos

Una vez validado el sistema Q&A:

1. **Punto 4**: Crear interfaz web con Streamlit o Gradio
2. **Punto 5**: Documentar el proceso y crear pruebas exhaustivas
3. **Módulos 2+**: Integrar con sistemas más avanzados (RAG, multi-agent, etc.)

## Documentación Adicional

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

---

**Estado**: ✅ Punto 3 - Sistema Q&A Implementado  
**Próximo**: 📋 Punto 4 - Interfaz de Usuario (Streamlit/Gradio)
