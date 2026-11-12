from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

KNOWLEDGE_FILE = "data/processed/fil_chunks.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "math_notes"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"#"Qwen/Qwen3-Reranker-0.6B"

LLM_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "deepseek-flash"
RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5