import logging

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from fastemb_langchain_adapter import FastEmbLangChainAdapter
from json_notes_loader import load_documents

KNOWLEDGE_FILE = "data/processed/fil_ag.jsonl"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "math_notes"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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