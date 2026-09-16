from pathlib import Path
import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from fastemb_langchain_adapter import FastEmbLangChainAdapter
from json_notes_loader import load_documents

from config import (
    LOG_DIR,
    EMBEDDING_MODEL,
    QDRANT_URL,
    KNOWLEDGE_FILE,
    COLLECTION_NAME
)

logger = logging.getLogger(__name__)

def build_index():
    logger.info(f"Loading documents form {KNOWLEDGE_FILE}")
    documents = load_documents(KNOWLEDGE_FILE)
    logger.info(f"Loaded {len(documents)} documents")

    logger.info(f"Loading embedding model {EMBEDDING_MODEL}")
    embeddings = FastEmbLangChainAdapter(EMBEDDING_MODEL)
    logger.info("Embedding model loaded")

    logger.info(f"Connecting to Qdrant database on url {QDRANT_URL}")
    client = QdrantClient(QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        logger.info("Found collection with same name")
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Old version collection {COLLECTION_NAME} deleted")

    logger.info(f"Creating collection {COLLECTION_NAME} with embeddings")
    vector_store = QdrantVectorStore.from_documents(documents, embedding=embeddings, url=QDRANT_URL, collection_name=COLLECTION_NAME)
    logger.info("Vector store created")

if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_DIR / "build_index.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
        force=True,
    )
    build_index()