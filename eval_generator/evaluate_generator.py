
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "shared"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
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
    answer_relevancy = AnswerRelevancyMetric(threshold=THRESHOLD, include_reason=True, model=judge)
    faithfulness = FaithfulnessMetric(threshold=THRESHOLD, include_reason=True, model=judge)

    results = evaluate(
        test_cases,
        metrics=[answer_relevancy, faithfulness],
        async_config=AsyncConfig(max_concurrent=1, throttle_value=8),
        cache_config=CacheConfig(write_cache=False),
    )
    return results


if __name__ == "__main__":
    main()