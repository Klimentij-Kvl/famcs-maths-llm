from langsmith import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT, ANSWER_RELEVANCE_PROMPT

from rag import eval_target


DATASET_NAME = "llm-math-100"
EXPERIMENT_PREFIX = "math-rag-dense"

correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model=""
)

results = evaluate(
    eval_target,
    data=DATASET_NAME,
    experiment_prefix=EXPERIMENT_PREFIX,
    max_concurrency=2,
)

print(results)