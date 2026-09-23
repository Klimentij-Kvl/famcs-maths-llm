from langsmith import evaluate

from rag import eval_target


DATASET_NAME = "llm-math-100"


results = evaluate(
    eval_target,
    data=DATASET_NAME,
    experiment_prefix="math-rag-baseline",
    max_concurrency=2,
)

print(results)