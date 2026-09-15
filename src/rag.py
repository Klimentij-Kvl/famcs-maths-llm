from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore

from build_index import (
    FastEmbLangChainAdapter,
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

LLM_MODEL = "qwen2.5:7b"
TOP_K = 5

embeddings = FastEmbLangChainAdapter(EMBEDDING_MODEL)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOllama(
    model=LLM_MODEL, 
    temperature=0,
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

def answer(question: str):
    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response, docs

