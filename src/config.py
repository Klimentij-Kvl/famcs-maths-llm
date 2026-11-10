from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

KNOWLEDGE_FILE = "data/processed/fil_chunks.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "math_notes"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"

LLM_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-flash"
TOP_K = 5