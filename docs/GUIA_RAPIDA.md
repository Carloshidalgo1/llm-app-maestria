# 🚀 Guía Rápida: Sistema Q&A (Punto 3)

## Resumen

Se ha implementado completamente el **Punto 3: Construcción del Aplicativo** con:

- ✅ **Script Principal**: `src/carnicos_kb/qa_system.py`
- ✅ **Knowledge Loader**: `src/carnicos_kb/knowledge_loader.py`
- ✅ **Base de Conocimiento**: Corpus consolidado desde punto 2 (36 archivos .md)
- ✅ **LLM Framework**: LangChain + OpenAI API
- ✅ **Prompt Engineering**: Robusto para evitar alucinaciones
- ✅ **Interfaz**: Chat interactivo + API programática

---

## ⚡ Inicio Rápido (5 minutos)

### 1️⃣ Obtener API Key de OpenAI
```
https://platform.openai.com/account/api-keys
→ Crear nueva key → Copiar clave
```

### 2️⃣ Configurar .env
```bash
# En el archivo .env, reemplaza:
OPENAI_API_KEY=sk_test_tu_api_key_aqui

# Por tu clave real:
OPENAI_API_KEY=sk-proj-xxxxx
```

### 3️⃣ Instalar dependencias
```bash
# Con uv (recomendado)
make sync

# O con pip
pip install langchain langchain-openai openai python-dotenv
```

### 4️⃣ Ejecutar el sistema Q&A
```bash
# Chat interactivo
make qa

# O pregunta única
python -c "from src.carnicos_kb.qa_system import CarnicosQASystem; qa = CarnicosQASystem(); print(qa.answer('¿Cuál es la misión?'))"
```

---

## 📋 Archivos Creados

### Scripts Python
- `src/carnicos_kb/qa_system.py` - Sistema Q&A con LangChain
- `src/carnicos_kb/knowledge_loader.py` - Cargador de corpus
- `example_qa.py` - Ejemplo de uso

### Configuración
- `.env` - Variables de entorno (actualizado con OPENAI_API_KEY)
- `pyproject.toml` - Dependencias (langchain, langchain-openai, openai)
- `Makefile` - Comando `make qa`

### Documentación
- `docs/QA_SYSTEM.md` - Documentación completa
- `docs/GUIA_RAPIDA.md` - Este archivo

---

## 🤖 Características del Sistema

### ✨ Prompt Engineering Robusto

```python
# El sistema implementa:
1. Base de conocimiento consolidada en el prompt de sistema
2. Instrucciones explícitas para evitar alucinaciones
3. Citas de fuentes
4. Temperatura baja (0.2) para precisión
5. Max tokens limitados para respuestas concisas
```

### 🎯 Respuestas Precisas

El sistema responde:
- **Solo** con información verificada del corpus
- Si la pregunta no está en la base de conocimiento, lo admite
- Con citas de la fuente del documento
- De forma profesional y clara

### 📊 Estadísticas Automáticas

Al inicializar, muestra:
```
📊 Estadísticas de la base de conocimiento:
  • Caracteres totales: 54,234
  • Palabras totales: 8,942
  • Párrafos totales: 142
```

---

## 💻 Uso Programático

### Opción 1: Chat Interactivo
```python
from src.carnicos_kb.qa_system import CarnicosQASystem

qa = CarnicosQASystem()
qa.interactive_chat()
```

### Opción 2: Pregunta Individual
```python
from src.carnicos_kb.qa_system import CarnicosQASystem

qa = CarnicosQASystem()
respuesta = qa.answer("¿Cuál es tu nombre?")
print(respuesta)
```

### Opción 3: Línea de Comandos
```bash
python -m src.carnicos_kb.qa_system "Tu pregunta aquí"
```

---

## 🧪 Pruebas Sugeridas (20+ preguntas)

### Categoría: Información Institucional
1. ¿Cuál es la misión de Alimentos Cárnicos?
2. ¿Cuál es la visión?
3. ¿Cuándo fue fundada?
4. ¿Dónde está ubicada?
5. ¿Quiénes son los líderes?

### Categoría: Productos
6. ¿Qué productos ofrecen?
7. ¿Cuáles son características de los productos?
8. ¿Hay productos premium?
9. ¿Tienen carnes frescas?

### Categoría: Contacto
10. ¿Cuál es el teléfono?
11. ¿Cuál es el email?
12. ¿Cuál es la dirección?
13. ¿Cuál es el horario?
14. ¿Cómo agendar una cita?

### Categoría: Sostenibilidad
15. ¿Cuál es el compromiso ambiental?
16. ¿Tienen certificaciones?
17. ¿Cómo manejan el bienestar animal?
18. ¿Usan prácticas sostenibles?

### Categoría: Preguntas Fuera de Alcance
19. ¿Cuál es la capital de Francia?
20. ¿Cómo se hace una pizza?

---

## 🔧 Configuración Avanzada

### Cambiar Modelo
```bash
# En .env
OPENAI_MODEL=gpt-4o  # Para mejor calidad
# o
OPENAI_MODEL=gpt-3.5-turbo  # Para menor costo
```

### Ajustar Temperatura
```bash
# En .env (0 = determinístico, 1 = creativo)
OPENAI_TEMPERATURE=0.5  # Aumentar creatividad
# o
OPENAI_TEMPERATURE=0.0  # Máxima precisión
```

### Aumentar Max Tokens
```bash
# En .env
OPENAI_MAX_TOKENS=2000  # Respuestas más largas
```

---

## ⚠️ Troubleshooting

### Error: "OPENAI_API_KEY no está configurada"
```bash
# 1. Verifica que .env existe
# 2. Verifica que tiene la clave correcta
# 3. Reinicia la terminal
# 4. Asegúrate de que no hay espacios en blanco
```

### Error: "No hay archivos .md"
```bash
# Necesitas generar el corpus primero
make scrape
# o
python -m src.carnicos_kb.scraper
```

### Respuestas vagas
```bash
# 1. Incrementa la temperatura en .env
# 2. Verifica que la información está en el corpus
# 3. Reformula la pregunta con más contexto
```

---

## 📈 Métricas de Éxito

El sistema es exitoso si:

- ✅ Carga sin errores de API key
- ✅ Responde preguntas sobre la empresa
- ✅ Rechaza preguntas fuera del alcance
- ✅ Las respuestas citan fuentes
- ✅ Responde en menos de 5 segundos
- ✅ Accuracy > 80% en 20 pruebas

---

## 🎯 Próximos Pasos

1. **Validación**: Realiza las 20+ pruebas sugeridas
2. **Documentación**: Registro de resultados en `Requisitos.txt`
3. **Punto 4**: Crear interfaz web (Streamlit/Gradio)
4. **Punto 5**: Pruebas exhaustivas y presentación

---

## 📚 Referencias

- [LangChain Docs](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)

---

**Estado**: ✅ Punto 3 Completo  
**Próximo**: Punto 4 (Interfaz Web)
