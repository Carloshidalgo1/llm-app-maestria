# RAG vectorial con `text-embedding-3-small`

Este documento acompana el script independiente:

```text
src/carnicos_kb/rag_index_builder.py
```

El objetivo es construir un indice local para RAG vectorial. El agente actual
puede seguir funcionando con su recuperador documental, pero este script deja
preparada la capa de embeddings que se espera en un RAG vectorial clasico.

## Flujo del script

```text
base_conocimiento_chunks.md
  |
  v
parsear chunks C0001, C0002, ...
  |
  v
preparar texto: id + titulo + fuente + contenido
  |
  v
OpenAIEmbeddings(model="text-embedding-3-small")
  |
  v
data/vector_index/carnicos_rag_index.json
o
data/vector_index/chroma/
```

## Por que `text-embedding-3-small`

- El corpus tiene pocos cientos de chunks, no millones de documentos.
- Es suficiente para busqueda semantica sobre preguntas frecuentes,
  sostenibilidad, productos, marcas y datos corporativos.
- Tiene menor costo que `text-embedding-3-large`.
- Permite demostrar RAG vectorial real sin introducir infraestructura externa.

## Comandos

Validar que los chunks se leen correctamente, sin llamar a OpenAI:

```powershell
make rag-index-dry-run
```

Construir el indice vectorial real:

```powershell
make rag-index
```

Construir la base vectorial local en Chroma:

```powershell
make rag-index-chroma
```

Comando directo equivalente:

```powershell
uv run carnicos-build-rag --chunks-path data/processed/base_conocimiento_chunks.md --output data/vector_index/carnicos_rag_index.json --embedding-model text-embedding-3-small
```

Comando directo con Chroma:

```powershell
uv run carnicos-build-rag --chunks-path data/processed/base_conocimiento_chunks.md --embedding-model text-embedding-3-small --vector-store chroma --chroma-dir data/vector_index/chroma --chroma-collection carnicos_rag
```

## Variables de entorno

```env
OPENAI_API_KEY=sk-proj-tu_clave_real
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CARNICOS_KNOWLEDGE_PATH=data/processed/base_conocimiento_chunks.md
CARNICOS_RAG_INDEX_PATH=data/vector_index/carnicos_rag_index.json
RAG_EMBEDDING_BATCH_SIZE=64
RAG_VECTOR_STORE=json
CHROMA_PERSIST_DIRECTORY=data/vector_index/chroma
CHROMA_COLLECTION_NAME=carnicos_rag
CARNICOS_DOCUMENTAL_RETRIEVER=lexical
```

## Archivo generado

El JSON generado contiene:

- `metadata`: fecha de construccion, modelo, fuente, cantidad de chunks y
  dimensiones del vector.
- `records`: lista de chunks con `chunk_id`, titulo, fuente, texto y embedding.

Ejemplo conceptual:

```json
{
  "metadata": {
    "embedding_model": "text-embedding-3-small",
    "chunk_count": 329,
    "embedding_dimensions": 1536
  },
  "records": [
    {
      "chunk_id": "C0001",
      "title": "alimentoscarnicos.com.co.md | Nuestra Historia",
      "source": "data/processed/dataset_carnicos/alimentoscarnicos.com.co.md",
      "text": "...",
      "embedding": [0.0123, -0.0456]
    }
  ]
}
```

## Base Chroma generada

Cuando se usa `--vector-store chroma`, los chunks se guardan en una coleccion
persistente local:

```text
data/vector_index/chroma/
```

Cada documento en Chroma conserva:

- contenido preparado para embedding: id, titulo, fuente y texto del chunk;
- metadatos: `chunk_id`, `title`, `source` y `source_path`;
- embedding calculado con `text-embedding-3-small`.

Para que el agente use Chroma como recuperador documental, despues de construir
la base se puede configurar:

```env
CARNICOS_DOCUMENTAL_RETRIEVER=chroma
```

## Como sustentarlo

Explicacion corta:

> Primero convertimos cada chunk documental en un vector numerico usando
> `text-embedding-3-small`. Luego guardamos esos vectores con su texto y fuente.
> En una consulta RAG vectorial, la pregunta tambien se convierte en vector y se
> comparan similitudes para recuperar los chunks mas cercanos antes de llamar al
> LLM.

Importante: la herramienta de datos estructurados no debe usar embeddings. Datos
como NIT, telefonos y sedes se recuperan de forma determinista desde JSON.
