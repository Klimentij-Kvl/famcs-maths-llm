from langsmith import evaluate
from langchain_ollama import ChatOllama
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from rag import eval_target
from config import (
    TOP_K, 
    LLM_MODEL, 
    EMBEDDING_MODEL,
    JUDGE_MODEL,
)

DATASET_NAME = "llm-math-small"

judge_model = ChatOllama(
    model=JUDGE_MODEL,
    temperature=0,
)

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=judge_model,
    feedback_key="correctness",
)

helpfulness_judge = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    judge=judge_model,
    feedback_key="helpfulness"
)
groundedness_judge = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    judge=judge_model,
    feedback_key="groundedness",
)
retrieval_relevance_judge = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    judge=judge_model,
    feedback_key="retrieval_relevance",
)

print(correctness_judge(
    inputs="What is 2 + 2?",
    outputs="The answer is 4.",
    reference_outputs="4",
))

print(helpfulness_judge(
    inputs={"question": "What is 2 + 2?"},
    outputs={"answer": "The answer is 4."},
))

print(groundedness_judge(
    context={"documents": ["2 + 2 = 4"]},
    outputs={"answer": "The answer is 4."},
))

print(retrieval_relevance_judge(
    inputs={"question": "What is 2 + 2?"},
    context={"documents": ["2 + 2 = 4"]},
    outputs=''
))