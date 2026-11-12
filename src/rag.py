import logging
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from sentence_transformers import CrossEncoder
from langsmith import traceable
import torch

from fastemb_langchain_adapter import FastEmbLangChainAdapter
from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    RERANKER_MODEL,
    LOG_DIR,
    LLM_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

load_dotenv()

logger = logging.getLogger(__name__)

embeddings = FastEmbLangChainAdapter(EMBEDDING_MODEL)
sparse_embeddings = FastEmbedSparse(model_name=SPARSE_EMBEDDING_MODEL)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    retrieval_mode=RetrievalMode.HYBRID,
)

retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})

reranker = CrossEncoder(
    RERANKER_MODEL,
    trust_remote_code=True,
    max_length=512,
    device=DEVICE,
)

llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
    metadata={
        "ls_model_name": LLM_MODEL,
    }
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Ты — учитель математики.

Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст.

Правила:

1. Не выдумывай теоремы и формулы. 
2. Если ответа нет в контексте, честно скажи:
   "В предоставленных конспектах я не нашёл ответа."
3. Когда решаешь задачу, объясняй ход решения последовательно.
4. Формулы сохраняй в LaTeX.
5. Если в контексте есть доказательство подходящей теоремы,
   используй его как основу объяснения.
6. Не утверждай, что в конспекте сказано то, чего там нет.
7. Не добавляй никаких источников в ответ.
8. Отвечай только на русском языке.

Контекст:

{context}
""",
    ),
    (
        "human",
        "{question}",
    ),
])

def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        header = (
            f"[Источник {i}]\n"
            f"Тип: {metadata.get('type')}\n"
            f"Раздел: {metadata.get('section')}\n"
            f"Файл: {metadata.get('source_file')}\n"
            f"Строки: "
            f"{metadata.get('start_line')}-"
            f"{metadata.get('end_line')}"
        )
        formatted.append(
            header
            + "\n\n"
            + doc.page_content
        )
    return "\n\n---\n\n".join(
        formatted
    )

def format_reranker_doc(doc):
    metadata = doc.metadata

    has_proof = metadata.get(
        "contains_proof",
        metadata.get("contatins_proof")
    )

    return (
        f"Type: {metadata.get('type', '')}\n"
        f"Chapter: {metadata.get('chapter', '')}\n"
        f"Section: {metadata.get('section', '')}\n"
        f"Contains proof: {has_proof}\n\n"
        f"Content:\n{doc.page_content}"
    )


def rerank_docs(query: str, docs: list):
    if not docs:
        return []

    pairs = [
        (query, format_reranker_doc(doc))
        for doc in docs
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [
        doc
        for doc, score in ranked_docs[:RERANK_TOP_K]
    ]

@traceable(name="math_rag")
def answer(query: str):
    candidate_docs = retriever.invoke(query)

    docs = rerank_docs(query, candidate_docs)

    context = format_docs(docs)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response, docs

def eval_target(inputs: dict) -> dict:
    query = inputs["query"]

    response, docs = answer(query)

    retrieved_chunk_ids = [
        str(doc.metadata["chunk_id"])
        for doc in docs
        if doc.metadata.get("chunk_id") is not None
    ]

    context = {
        doc.page_content
        for doc in docs
    }

    return {
        "answer": response,
        "context": context,
        "retrieved_chunk_ids": retrieved_chunk_ids,
    }

def main():
    logging.basicConfig(
        filename=LOG_DIR / "rag.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8"
    )

    print("\n" + 
          "=" * 60 + 
          "\nMATH RAG\n" + 
          "=" * 60 + 
          "\nВведите запрос. \nДля выхода: exit\n")

    while True:
        question = input(">>> ").strip()

        if question.lower() in("exit", "quit", "q"):
            break

        if not question:
            continue

        response, docs = answer(question)

        for doc in docs:
            logger.info(doc)

        print()
        print(response)

if __name__ == "__main__":
    main()