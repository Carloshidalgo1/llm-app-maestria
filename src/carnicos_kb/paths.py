from pathlib import Path


DEFAULT_RAW_PDF_DIR = Path("data/raw/pdfs")
DEFAULT_DATASET_DIR = Path("data/processed/dataset_carnicos")
DEFAULT_CHUNKS_FILE = Path("data/processed/base_conocimiento_chunks.md")
DEFAULT_STRUCTURED_DATA_FILE = Path("data/structured/carnicos_structured_faq.json")
DEFAULT_POSTGRES_URL = "postgresql://user:password@localhost:5432/carnicos_kb"
DEFAULT_PG_COLLECTION = "carnicos_rag"
