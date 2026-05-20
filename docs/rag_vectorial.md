# RAG vectorial con `text-embedding-3-small`

Este documento acompana el script independiente:

```text
src/carnicos_kb/rag_index_builder.py
```

El objetivo es construir la capa de recuperacion semantica del agente usando
embeddings de OpenAI y un backend PGVector en PostgreSQL. En desarrollo, el
codigo tambien puede construir un `InMemoryVectorStore` para validaciones sin
persistencia.

## Flujo del script

```text
data/processed/dataset_carnicos/
  |
  v
RecursiveCharacterTextSplitter
  |
  v
Document(page_content, metadata)
  |
  v
OpenAIEmbeddings(model="text-embedding-3-small")
  |
  v
PostgreSQL / PGVector
```

## Por que `text-embedding-3-small`

- El corpus tiene pocos cientos de chunks, no millones de documentos.
- Es suficiente para busqueda semantica sobre preguntas frecuentes,
  sostenibilidad, productos, marcas y datos corporativos.
- Tiene menor costo que `text-embedding-3-large`.
- Permite demostrar RAG vectorial real con una base persistente.

## Comandos

Validar lectura y chunking sin llamar a OpenAI ni escribir en PostgreSQL:

```powershell
make rag-index-dry-run
```

Construir el indice vectorial persistente:

```powershell
make rag-index
```

Comando directo equivalente:

```powershell
uv run carnicos-build-rag --dataset-dir data/processed/dataset_carnicos --embedding-model text-embedding-3-small --collection carnicos_rag
```

## Variables de entorno

```env
OPENAI_API_KEY=sk-proj-tu_clave_real
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql://user:password@localhost:5432/carnicos_kb
PG_COLLECTION_NAME=carnicos_rag
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
```

## Backend PGVector

`rag_index_builder.py` crea documentos LangChain desde los Markdown procesados,
calcula embeddings y los almacena en la coleccion configurada por
`PG_COLLECTION_NAME` o `--collection`.

Cada documento conserva metadatos trazables:

- `source`: ruta del Markdown origen;
- `title`: nombre base del archivo fuente.

El agente usa `PGVectorRetriever` cuando `DATABASE_URL` esta configurada. Si no
hay `DATABASE_URL`, usa `InMemoryVectorStore` como fallback de desarrollo.

## Como sustentarlo

Explicacion corta:

> Primero dividimos cada documento Markdown en fragmentos con
> `RecursiveCharacterTextSplitter`. Luego convertimos esos fragmentos en
> vectores usando `text-embedding-3-small` y los guardamos en PGVector. En una
> consulta RAG, la pregunta tambien se convierte en vector y se comparan
> similitudes para recuperar los fragmentos mas cercanos antes de llamar al LLM.

Importante: la herramienta de datos estructurados no usa embeddings. Datos como
NIT, telefonos y sedes se recuperan de forma deterministica desde JSON.
