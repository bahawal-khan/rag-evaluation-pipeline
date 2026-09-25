
import os
import sys
import json
from pathlib import Path

# DeepEval ka apna internal per-task timeout disable kar rahe hain — kyunke
# humara apna retry mechanism (groq_judge.py) already safe/bounded hai
# (max 6 attempts ke baad khud ruk jata hai), lekin Groq ke rate-limit waits
# (12s, 24s, 36s...) kabhi kabhi DeepEval ke default timeout se zyada ho
# jate hain, jo poora evaluation cancel kar deta tha.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from retriever import Retriever
from generator import build_generator, generate_answer
from groq_judge import GroqJudge
from paths import GOLDEN_DATASET_PATH

TOP_K = 3
THRESHOLD = 0.75


def load_golden_dataset():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(golden_data, retriever: Retriever, llm):
    test_cases = []
    for item in golden_data:
        retrieved_chunks = retriever.retrieve(item["input"])
        actual_output = generate_answer(llm, item["input"], retrieved_chunks)

        test_case = LLMTestCase(
            input=item["input"],
            actual_output=actual_output,
            expected_output=item["expected_output"],
            retrieval_context=retrieved_chunks,
        )
        test_cases.append(test_case)
        print(f"\nQ: {item['input']}\nA: {actual_output}\n{'-'*60}")

    return test_cases


def main():
    golden_data = load_golden_dataset()
    retriever = Retriever(top_k=TOP_K)
    llm = build_generator()

    test_cases = build_test_cases(golden_data, retriever, llm)

    judge = GroqJudge()
    metrics = [
        ContextualPrecisionMetric(threshold=THRESHOLD, include_reason=True, model=judge),
        ContextualRecallMetric(threshold=THRESHOLD, include_reason=True, model=judge),
        AnswerRelevancyMetric(threshold=THRESHOLD, include_reason=True, model=judge),
        FaithfulnessMetric(threshold=THRESHOLD, include_reason=True, model=judge),
    ]

    # 4 metrics x 10 questions = zyada Groq calls, isliye throttle thora barha diya
    results = evaluate(
        test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=1, throttle_value=10),
        cache_config=CacheConfig(write_cache=False),
    )
    return results


if __name__ == "__main__":
    main()