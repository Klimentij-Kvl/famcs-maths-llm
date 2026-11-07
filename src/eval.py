from langsmith import evaluate
from langchain_ollama import ChatOllama
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
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
groundedness_judge = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    judge=judge_model,
    feedback_key="groundedness",
)
retrieval_judge = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    judge=judge_model,
    feedback_key="retrieval_relevance",
)

def recall(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = reference_outputs['chunks']
    if not reference:
        return {
            "key": "recall", 
            "score": 0.0
        }
    score = sum(s in retrieved for s in reference) / len(reference)
    return {
        "key": "recall", 
        "score": score
    }


def precision(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = reference_outputs['chunks']
    if not retrieved:
        return {
            "key": "precision", 
            "score": 0.0
        }
    score = sum(s in retrieved for s in reference) / len(retrieved)
    return {
        "key": "precision", 
        "score": score
    }


def hit_rate(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = reference_outputs['chunks']
    score = float(any(s in retrieved for s in reference))
    return {
        "key": "hit rate", 
        "score": score
    }

def mrr(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = reference_outputs['chunks']
    score = 0.0
    for idx, chunk in enumerate(retrieved):
        if chunk in reference:
            score = 1.0 / (idx + 1)
    return {
        "key": "mrr", 
        "score": score
    }

results = evaluate(
    eval_target,
    data=DATASET_NAME,
    evaluators=[
        recall, 
        precision,
        hit_rate,
        mrr
    ],
    experiment_prefix="famcs-maths-sparse",
    description="Testing the sparse search",
    metadata={
        "Retrieval Mode": "Sparse",
        "LLM": LLM_MODEL,
        "Embedding model": EMBEDDING_MODEL,
        "Judge Model": JUDGE_MODEL,
    }
)

print(results)