
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

# shared/ folder ko import-path mein add karna, taake retriever.py,
# groq_judge.py, paths.py yahan se import ho sakein
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from retriever import Retriever
from groq_judge import GroqJudge
from paths import GOLDEN_DATASET_PATH

TOP_K = 3
THRESHOLD = 0.75


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(golden_data, retriever: Retriever):
    test_cases = []
    for item in golden_data:
        retrieved_chunks = retriever.retrieve(item["input"])
        test_case = LLMTestCase(
            input=item["input"],
            actual_output="N/A - retriever-only evaluation, generator abhi test nahi ho raha",
            expected_output=item["expected_output"],
            retrieval_context=retrieved_chunks,
        )
        test_cases.append(test_case)
    return test_cases


def main():
    golden_data = load_golden_dataset()
    retriever = Retriever(top_k=TOP_K)
    test_cases = build_test_cases(golden_data, retriever)

    judge = GroqJudge()
    contextual_precision = ContextualPrecisionMetric(threshold=THRESHOLD, include_reason=True, model=judge)
    contextual_recall = ContextualRecallMetric(threshold=THRESHOLD, include_reason=True, model=judge)

    results = evaluate(
        test_cases,
        metrics=[contextual_precision, contextual_recall],
        async_config=AsyncConfig(max_concurrent=1, throttle_value=8),
        cache_config=CacheConfig(write_cache=False),
    )
    return results


if __name__ == "__main__":
    main()