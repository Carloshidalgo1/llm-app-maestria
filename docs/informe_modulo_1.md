# Informe formal - Modulo 1

## Creacion de la base de conocimiento semantico y sistema Q&A para Alimentos Carnicos S.A.S.

**Proyecto:** Carnicos KB  
**Empresa analizada:** Alimentos Carnicos S.A.S.  
**Modulo:** 1 - Base de conocimiento semantico y sistema Q&A  
**Interfaz de demostracion:** Streamlit  
**Repositorio:** Proyecto Python con scraping, extraccion de PDF, chunking, sistema Q&A e interfaz web.

---

## 1. Descripcion del problema

Alimentos Carnicos S.A.S. cuenta con informacion publica distribuida en diferentes fuentes digitales: sitio web corporativo, paginas de marcas, secciones institucionales, documentos PDF, politicas, contenidos de sostenibilidad, preguntas frecuentes, formularios y canales de atencion. Para un usuario que interactua por primera vez con la empresa, consultar esta informacion puede requerir navegar varias paginas o documentos, lo que dificulta obtener respuestas rapidas, claras y consistentes.

El problema identificado es la **necesidad de un canal de comunicacion automatizado y preciso para Alimentos Carnicos S.A.S.**, capaz de responder preguntas frecuentes de primer contacto sobre informacion institucional, productos, marcas, ubicacion, canales de atencion, PQRS, procesos basicos, sostenibilidad y tratamiento de datos personales.

El sistema requerido debe apoyar la consulta inicial de consumidores, clientes institucionales, interesados comerciales y usuarios que buscan orientacion basica. No debe reemplazar canales oficiales ni realizar procesos transaccionales reales, sino facilitar el acceso a informacion publica previamente recopilada y estructurada.

---

## 2. Planteamiento de la solucion

La solucion implementada consiste en la construccion de una base de conocimiento semantica y un sistema basico de preguntas y respuestas (Q&A) que utiliza un modelo de lenguaje a traves de API, orquestado con LangChain. El sistema toma el contenido limpio extraido de las fuentes publicas y lo consolida como contexto dentro del prompt de sistema para responder preguntas del usuario.

### Aclaracion metodologica sobre RAG

El requisito del punto 3 indica expresamente que esta etapa **no es un RAG**. Por esta razon, la implementacion actual no usa embeddings, busqueda semantica ni base de datos vectorial en tiempo de consulta. En su lugar, utiliza una estrategia de **contexto consolidado en el prompt de sistema**, donde la base de conocimiento segmentada funciona como memoria inicial del asistente.

Sin embargo, el punto 5 solicita explicar la solucion como nucleo para un futuro sistema basado en RAG. En este informe se entiende RAG como una **evolucion del modulo 2**, no como una caracteristica ya implementada en el modulo 1. La arquitectura actual deja preparados los insumos necesarios para esa evolucion: documentos limpios, chunks semanticos, fuentes trazables y un sistema Q&A que ya separa base de conocimiento, prompt e interfaz.

### Componentes principales

- **Base de conocimiento:** archivos Markdown en `data/processed/dataset_carnicos` y archivo consolidado `data/processed/base_conocimiento_chunks.md`.
- **Extraccion web:** script `src/carnicos_kb/scraper.py`, usando `requests`, `BeautifulSoup` y `trafilatura`.
- **Extraccion de PDF:** scripts `src/carnicos_kb/pdf_extractor.py` y `src/carnicos_kb/pdf_text_extractor.py`.
- **Chunking:** script `src/carnicos_kb/chunking.py`, con limpieza, division por encabezados y control de longitud.
- **Sistema Q&A:** clase `CarnicosQASystem` en `src/carnicos_kb/qa_system.py`.
- **Interfaz web:** app Streamlit en `src/carnicos_kb/streamlit_app.py`.

---

## 3. Preparacion de los datos

### 3.1 Fuentes de informacion

La base de conocimiento se construyo a partir de informacion publica asociada con Alimentos Carnicos S.A.S. y documentos relacionados. Las principales fuentes procesadas se almacenan en:

- `data/processed/dataset_carnicos/informacion_basica.md`
- `data/processed/dataset_carnicos/contacto.md`
- `data/processed/dataset_carnicos/preguntas-frecuentes.md`
- `data/processed/dataset_carnicos/carnes-frescas.md`
- `data/processed/dataset_carnicos/nuestra-historia.md`
- `data/processed/dataset_carnicos/nuestro-compromiso.md`
- `data/processed/dataset_carnicos/bienestar-animal.md`
- `data/processed/dataset_carnicos/politica-de-uso.md`
- `data/processed/dataset_carnicos/Politica_habeasdata.md`
- `data/processed/dataset_carnicos/Informe-de-sostenibilidad-resumen-grupo-nutresa.md`

Tambien se incluyeron documentos PDF almacenados en `data/raw/pdfs`, convertidos posteriormente a Markdown para integrarlos a la base.

### 3.2 Extraccion por web scraping

El scraping se realiza desde el sitemap configurado en `.env` o desde el valor por defecto:

```text
https://alimentoscarnicos.com.co/wp-sitemap-posts-page-1.xml
```

El script `scraper.py` ejecuta el siguiente flujo:

1. Descarga el sitemap XML usando `requests`.
2. Extrae las URLs con `BeautifulSoup`.
3. Visita cada pagina con una sesion HTTP.
4. Extrae texto util con `trafilatura`.
5. Guarda cada pagina como archivo Markdown en `data/processed/dataset_carnicos`.
6. Aplica una pausa configurable entre solicitudes para evitar carga excesiva sobre el sitio fuente.

Comando asociado:

```powershell
make scrape
```

### 3.3 Extraccion de PDFs

El proyecto permite dos estrategias:

- **Extraccion estructurada con Docling:** `make pdf`
- **Extraccion rapida de texto plano con PyMuPDF:** `make pdf-fast`

Los archivos PDF se leen desde `data/raw/pdfs` y sus salidas se escriben como Markdown en `data/processed/dataset_carnicos`. Esto permite integrar informes, politicas y documentos publicos a la misma base textual usada por el modelo.

### 3.4 Limpieza y segmentacion

La limpieza y segmentacion se realiza en `chunking.py`. El proceso incluye:

- Eliminacion de marcas de fuente repetidas.
- Eliminacion de comentarios HTML.
- Eliminacion de marcas de citas.
- Normalizacion de saltos de linea.
- Division por encabezados Markdown.
- Filtrado de bloques sin contenido util.
- Division de bloques largos segun un limite objetivo de caracteres.
- Renderizado final con metadatos de fuente, seccion y conteo de caracteres.

Comando asociado:

```powershell
make chunk
```

### 3.5 Artefactos generados

Al momento de elaborar este informe, el repositorio contiene:

| Artefacto | Cantidad / estado |
|---|---:|
| Archivos Markdown procesados | 43 |
| PDFs fuente | 6 |
| Chunks generados | 329 |
| Archivo consolidado de chunks | `data/processed/base_conocimiento_chunks.md` |
| Tamano del archivo consolidado | 554650 bytes |

---

## 4. Modelado

### 4.1 Framework de LLM

El proyecto utiliza **LangChain** como framework de orquestacion para construir los mensajes enviados al modelo y abstraer la comunicacion con el proveedor del LLM.

El modulo central es:

```text
src/carnicos_kb/qa_system.py
```

La clase principal es:

```python
CarnicosQASystem
```

### 4.2 Modelo de lenguaje

La implementacion usa un modelo privado via API de OpenAI mediante `langchain-openai`. El modelo es configurable por variable de entorno:

```text
OPENAI_MODEL=gpt-5.4-nano
```

Los parametros principales tambien son configurables:

```text
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=1500
```

La temperatura baja se utiliza para favorecer respuestas mas deterministicas y reducir variaciones innecesarias en un caso de Q&A factual.

### 4.3 Diseno del prompt

El prompt de sistema cumple cuatro funciones principales:

1. Define la identidad del asistente como experto en Alimentos Carnicos S.A.S.
2. Inserta la base de conocimiento verificada como contexto.
3. Ordena responder solo con informacion documentada.
4. Establece reglas para evitar alucinaciones, mantener el alcance y responder con tono formal.

Reglas relevantes del prompt:

- Usar unicamente la base de conocimiento cargada.
- No inventar fechas, telefonos, precios, horarios ni procesos.
- Indicar explicitamente cuando no exista informacion verificada.
- Recomendar verificacion por canales oficiales cuando el dato pueda cambiar.
- Mantener respuestas breves o estructuradas segun el tipo de pregunta.

### 4.4 Embeddings y base vectorial

En el modulo 1 no se implementan embeddings ni base de datos vectorial porque el requisito del punto 3 especifica que el aplicativo **no es un RAG**. La base de conocimiento se incorpora completa como contexto del prompt de sistema.

Para el modulo 2, la arquitectura recomendada es evolucionar hacia RAG con los siguientes componentes:

- **Modelo de embeddings:** un modelo de embeddings compatible con el proveedor elegido para convertir cada chunk en vectores.
- **Base vectorial:** Chroma, FAISS u otra base vectorial local o gestionada.
- **Retriever:** busqueda semantica de los chunks mas relevantes para cada pregunta.
- **LLM:** el mismo proveedor configurado por API, ajustado segun costo, latencia y calidad.
- **Prompt RAG:** instrucciones de respuesta que incluyan solo los fragmentos recuperados y sus fuentes.

Esta evolucion reduciria el tamano del prompt, mejoraria escalabilidad y permitiria usar bases de conocimiento mas grandes.

---

## 5. Resultados

### 5.1 Interfaz de demostracion

La interfaz web fue desarrollada en Streamlit y permite:

- Escribir preguntas en lenguaje natural.
- Visualizar respuestas generadas por el sistema.
- Consultar la definicion del alcance.
- Ver una guia rapida.
- Inspeccionar el estado de la base de conocimiento.
- Configurar modelo, temperatura y maximo de tokens desde la barra lateral.

Comando de ejecucion:

```powershell
make app
```

Ruta principal:

```text
src/carnicos_kb/streamlit_app.py
```

### 5.2 Pruebas tecnicas del repositorio

El repositorio incluye pruebas unitarias para validar el proceso de chunking:

```powershell
make test
```

Estas pruebas verifican:

- Limpieza de ruido textual.
- Division por encabezados.
- Construccion de chunks desde archivos Markdown.
- Renderizado del archivo consolidado con metadatos.

### 5.3 Matriz de validación funcional

La siguiente matriz recoge las 20 preguntas definidas en el alcance del proyecto. Deben ejecutarse en la interfaz Streamlit o por consola para registrar la respuesta exacta del modelo durante la sustentación o validación final.

| # | Pregunta | Respuesta esperada resumida | Criterio de aceptación |
|---:|---|---|---|
| 1 | ¿Qué empresa es Alimentos Cárnicos S.A.S.? | Alimentos Cárnicos S.A.S. es una empresa colombiana del Grupo Nutresa dedicada al procesamiento y conservación de carne y productos cárnicos.<br><br>También realiza actividades como comercio al por mayor de productos alimenticios y comercio al por menor de carnes en establecimientos especializados.<br><br>**Fuente:** informacion_basica.md (sección 1.4) | Identifica razón social y actividad sin inventar datos. |
| 2 | ¿A qué grupo empresarial pertenece Alimentos Cárnicos? | Alimentos Cárnicos S.A.S. pertenece al Grupo Empresarial Nutresa.<br><br>**Fuente:** informacion_basica.md | Menciona el grupo correcto. |
| 3 | ¿Dónde está ubicada la sede principal? | Carrera 40 # 12A – 13, Yumbo, Valle del Cauca, Colombia.<br><br>**Fuente:** informacion_basica.md (sección 3.1) | Entrega dirección documentada y recomienda verificar si aplica. |
| 4 | ¿Cuál es la historia de Alimentos Cárnicos? | - 1935: Inicia con la Salsamentaría Suiza.<br>- 1960–2000s: consolidación del negocio.<br>- 2008: fusión de 7 empresas.<br>- Actualidad: operación con tecnología avanzada bajo Grupo Nutresa.<br><br>**Fuente:** nuestra-historia.md | Respuesta histórica coherente y sin extrapolaciones. |
| 5 | ¿Qué relación tiene Rica Rondo con Alimentos Cárnicos? | Rica Rondo hace parte del Negocio Cárnico de Grupo Nutresa y fue incluida en la fusión de 2008.<br><br>También operó en Zona Franca del Cauca (Puerto Tejada), con cierre en marzo de 2025.<br><br>**Fuentes:** nuestra-historia.md, informacion_basica.md | No confunde marca con razón social si la fuente no lo permite. |
| 6 | ¿Qué marcas hacen parte del portafolio mencionado? | Marcas del portafolio:<br>- Rica / Rica Rondo<br>- Cunit<br>- Suizo<br>- Americana<br>- Pietrán<br><br>Ranchera y Zenú pertenecen a Industria de Alimentos Zenú S.A.S.<br><br>**Fuente:** informacion_basica.md (2.1) | Lista marcas documentadas. |
| 7 | ¿Qué productos ofrece Cunit? | Categorías principales:<br>- Salchichón<br>- Chorizo<br>- Jamón<br>- Salchichas y tocinetas<br>- Mortadelas<br>- Butifarras<br><br>Incluye diferentes presentaciones comerciales.<br><br>**Fuente:** catálogo comercial | No inventa presentaciones no presentes. |
| 8 | ¿Qué productos ofrece Rica? | Categorías:<br>- Salchichas<br>- Salchichón<br>- Jamones<br>- Mortadelas<br>- Chorizos y hamburguesas<br>- Tocinetas<br>- Carnes frescas<br><br>**Fuente:** catálogo y carnes frescas | Respuesta comercial clara. |
| 9 | ¿Qué tipos de carnes frescas comercializa la empresa? | Categorías de cortes de res:<br>- Cortes de la rueda<br>- Cuarto delantero<br>- Cortes finos<br>- Lagartos<br>- Delantero robado<br>- Recortes 80/20<br><br>**Fuente:** carnes-frescas.md | Usa solo categorías presentes en la base. |
| 10 | ¿Qué cortes de res aparecen en el portafolio? | Cortes documentados:<br>- Tabla, extranjero, huevo de aldana, posta, muchacho<br>- Aguja, entretabla, pecho, sabaleta, paletero, falda, cogote<br>- Solomito, chata, solomo<br>- Lagarto delantero y trasero<br><br>**Fuente:** carnes-frescas.md | Enumera cortes o categorías documentadas. |
| 11 | ¿Cómo puedo comunicarme con servicio al cliente? | Canales:<br>- Línea nacional: 01 8000 52 73<br>- Teléfono Medellín: (60) (4) 470 52 22<br>- Web oficial<br><br>**Fuente:** contacto.md | Recomienda verificar canales oficiales. |
| 12 | ¿Cómo puedo radicar una PQRS? | Procedimiento:<br>1. Ingresar al sitio web.<br>2. Ir a sección PQRS.<br>3. Diligenciar formulario.<br>4. Adjuntar evidencias.<br>5. Conservar radicado.<br><br>También vía telefónica.<br><br>**Fuente:** informacion_basica.md | No simula radicación real. |
| 13 | ¿Cuáles son los puntos de venta propios mencionados? | Puntos registrados:<br>- Cali<br>- Bogotá (Parador Suizo)<br>- Barranquilla<br><br>**Fuente:** preguntas-frecuentes.md | Advierte posible actualización. |
| 14 | ¿Cómo puede un cliente solicitar una degustación? | - Cliente: contactar ejecutivo comercial.<br>- Persona natural: enviar carta con datos y descripción del evento.<br><br>Incluye datos personales y justificación.<br><br>**Fuente:** base de conocimiento | Orienta proceso sin inventar. |
| 15 | ¿La empresa realiza donaciones directamente? | No realiza donaciones directas.<br><br>Se canalizan a través de bancos de alimentos.<br><br>**Fuente:** preguntas frecuentes | Mantiene lenguaje neutral. |
| 16 | ¿Cómo puedo aplicar a empleo o práctica? | Procedimiento:<br>1. Ingresar a “Trabaje con nosotros”.<br>2. Registrar hoja de vida.<br>3. Actualizar información.<br><br>La empresa evalúa vacantes disponibles.<br><br>**Fuente:** preguntas frecuentes | No informa vacantes específicas. |
| 17 | ¿Qué requisitos existen para solicitar visita a planta? | Requisitos:<br>- Carta con datos completos<br>- Lista de visitantes<br>- Justificación<br>- Mínimo 30 personas<br>- Edad mínima: 7 años<br><br>**Fuente:** preguntas frecuentes | No inventa requisitos adicionales. |
| 18 | ¿Qué compromisos tiene sobre bienestar animal? | Compromisos:<br>- Meta cage-free 2027<br>- Bienestar porcino<br>- No clonación genética<br>- Control de medicamentos<br><br>**Fuente:** bienestar-animal.md | Alineado a sostenibilidad. |
| 19 | ¿Qué acciones sociales o comunitarias aparecen? | Acciones:<br>- Donaciones a bancos de alimentos<br>- Voluntariado<br>- Apoyo cultural<br>- Alimentando Talentos<br><br>**Fuente:** sostenibilidad | No exagera impacto. |
| 20 | ¿Qué derechos tengo sobre mis datos personales? | Derechos:<br>- Conocer, actualizar y rectificar<br>- Solicitar prueba<br>- Ser informado<br>- Presentar quejas<br>- Revocar autorización<br>- Acceder gratuitamente<br><br>**Fuente:** habeas data | Evita asesoría legal personalizada. |

### 5.4 Ejemplos de respuestas esperadas

**Pregunta:** Que empresa es Alimentos Carnicos S.A.S.?  
**Respuesta esperada:** Alimentos Carnicos S.A.S. es una empresa del sector carnico vinculada al Grupo Nutresa y al Negocio Carnico de Colombia. Su informacion publica la presenta como una organizacion dedicada al procesamiento y conservacion de carne y productos carnicos.  
**Analisis:** La respuesta debe ser breve, institucional y basada en la informacion consolidada.

**Pregunta:** Como puedo radicar una PQRS?  
**Respuesta esperada:** El asistente debe orientar al usuario hacia los canales documentados para PQRS y aclarar que no puede radicar solicitudes reales desde la interfaz. Si el canal puede cambiar, debe recomendar verificarlo en el sitio oficial o canales de atencion de la empresa.  
**Analisis:** La respuesta es correcta si orienta sin simular una transaccion.

**Pregunta:** Cual es el precio actualizado de una salchicha Rica?  
**Respuesta esperada:** El asistente debe indicar que no tiene informacion verificada sobre precios actualizados en la base de conocimiento y recomendar consultar canales oficiales o puntos de venta.  
**Analisis:** Esta pregunta esta fuera del alcance porque requiere informacion en tiempo real.

### 5.5 Calidad observada y limitaciones

La calidad esperada del sistema es adecuada para una primera fase de Q&A, siempre que las preguntas se mantengan dentro del alcance definido. El enfoque de contexto consolidado permite demostrar el comportamiento del asistente sin implementar todavia recuperacion semantica.

Limitaciones identificadas:

- El sistema no tiene acceso a informacion en tiempo real.
- No puede confirmar precios, promociones, inventario, horarios actualizados ni disponibilidad.
- No realiza transacciones, radicaciones ni seguimiento de solicitudes.
- El prompt puede crecer demasiado si la base aumenta; esto justifica migrar a RAG en el modulo 2.
- La precision final depende de la calidad del scraping, la limpieza y la cobertura de la base.
- La matriz de 20 preguntas debe diligenciarse con respuestas reales del modelo antes de la entrega final.

---

## 6. Presentacion y demostracion

La sustentacion debe realizarse en vivo, sin presentacion de PowerPoint, usando el repositorio y la interfaz Streamlit.

### Guion sugerido de 15 minutos

| Tiempo | Actividad |
|---:|---|
| 2 min | Explicar el problema y el alcance del asistente. |
| 3 min | Mostrar fuentes, scraping, PDFs y base de conocimiento. |
| 3 min | Explicar limpieza, chunking y archivo consolidado. |
| 3 min | Explicar Q&A, prompt engineering y restricciones anti-alucinacion. |
| 3 min | Demostrar la interfaz Streamlit con preguntas dentro y fuera del alcance. |
| 1 min | Cerrar con limitaciones y evolucion hacia RAG en modulo 2. |

### Comandos para la demostracion

```powershell
make sync
make chunk
make app
```

Si el puerto por defecto esta ocupado:

```powershell
make app STREAMLIT_PORT=8502
```

---

## 7. Conclusiones

El proyecto cumple la base tecnica del modulo 1: construye una base de conocimiento a partir de informacion publica, procesa documentos en Markdown, segmenta el contenido en chunks y ofrece un sistema Q&A funcional con una interfaz web en Streamlit.

Respecto al punto 5, el informe formal queda estructurado con las secciones solicitadas. Para cerrar completamente la evidencia de evaluacion, se recomienda ejecutar las 20 preguntas de validacion y registrar las respuestas exactas generadas por el modelo, junto con una calificacion de precision y coherencia.

La solucion actual no implementa RAG por restriccion del punto 3, pero deja lista la base documental y la arquitectura conceptual para incorporar embeddings, recuperacion semantica y una base vectorial en el siguiente modulo.
