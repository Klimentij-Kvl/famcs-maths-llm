from langsmith import evaluate
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from dotenv import load_dotenv

from rag import eval_target
from config import (
    TOP_K, 
    LLM_MODEL, 
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    JUDGE_MODEL,
)

DATASET_NAME = "llm-math-100"

load_dotenv()

judge_model = ChatDeepSeek(
    model=JUDGE_MODEL,
    temperature=0,
    max_retries=2,
    timeout=60,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
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

# RETRIEVER EVALUATION

def recall(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = set(reference_outputs['chunks'])
    if not reference:
        return {
            "key": "recall", 
            "score": 0.0,
        }
    score = sum(s in retrieved for s in reference) / len(reference)
    return {
        "key": "recall", 
        "score": score,
    }


def precision(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = set(reference_outputs['chunks'])
    if not retrieved:
        return {
            "key": "precision", 
            "score": 0.0,
        }
    score = sum(s in retrieved for s in reference) / len(retrieved)
    return {
        "key": "precision", 
        "score": score,
    }


def hit_rate(outputs: dict, reference_outputs: dict):
    retrieved = set(outputs['retrieved_chunk_ids'])
    reference = set(reference_outputs['chunks'])
    score = float(any(s in retrieved for s in reference))
    return {
        "key": "hit rate", 
        "score": score,
    }

def mrr(outputs: dict, reference_outputs: dict):
    retrieved = outputs["retrieved_chunk_ids"]
    reference = set(reference_outputs["chunks"])

    score = 0.0

    for rank, chunk in enumerate(retrieved, start=1):

        if chunk in reference:
            score = 1.0 / rank
            break

    return {
        "key": "mrr",
        "score": score,
    }


# GENERATION EVALUATION

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    return correctness_judge(
        inputs=inputs['query'],
        outputs=outputs['answer'],
        reference_outputs=reference_outputs['ground_truth_answer'],
    )

def helpfulness_evaluator(inputs: dict, outputs: dict):
    return helpfulness_judge(
        inputs=inputs['query'],
        outputs={"answer": outputs["answer"]},
    )

def groundedness_evaluator(outputs: dict):
    return groundedness_judge(
        context={"documents": outputs['context']},
        outputs={"answer": outputs['answer']},
    )

def retrieval_relevance_evaluator(inputs, outputs):
    return retrieval_relevance_judge(
        inputs=inputs['query'],
        context={"documents": outputs['context']},
        outputs=''
    )

results = evaluate(
    eval_target,
    data=DATASET_NAME,
    evaluators=[
        recall, 
        precision,
        hit_rate,
        mrr,
        correctness_evaluator,
        helpfulness_evaluator,
        groundedness_evaluator,
        retrieval_relevance_evaluator,
    ],
    experiment_prefix="famcs-maths",
    description="Testing the math rag",
    metadata={
        "Retrieval Mode": "Hybrid",
        "LLM": LLM_MODEL,
        "Embedding model": EMBEDDING_MODEL,
        "Sparse embedding model": SPARSE_EMBEDDING_MODEL,
        "Judge Model": JUDGE_MODEL,
        "Reranker": "NO",
        "top K": TOP_K,
    }
)

print(results)