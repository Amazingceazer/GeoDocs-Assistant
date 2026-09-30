"""Draft Q&A pairs from random chunks. REVIEW AND EDIT the output by hand.

Usage: python -m eval.gen_questions 40 > eval/qa_pairs.jsonl
"""
import json
import random
import sys

from app.config import settings
from app.llm import provider
from app.retrieval.store import _client

SYSTEM = (
    "Write ONE specific question that can be answered only from the passage. "
    "Do not mention 'the passage' or 'the document'. Output the question only."
)

# Add near-miss unanswerable questions about YOUR topic too (topic-adjacent but not in the docs).
UNANSWERABLE = [
    "What is the capital of Australia?",
    "How do I bake sourdough bread?",
    "What is the current price of Bitcoin?",
    "Who won the 2018 World Cup?",
    "Explain how transformers work in machine learning.",
]


def main(n: int) -> None:
    points, _ = _client().scroll(settings.collection, limit=500, with_payload=True)
    points = [p for p in points if len(p.payload["text"]) > 400]
    for p in random.sample(points, min(n, len(points))):
        q = provider.generate(SYSTEM, p.payload["text"]).strip()
        print(json.dumps({
            "question": q,
            "expected_source": p.payload["source"],
            "expected_pages": [p.payload["page"]],
            "answerable": True,
        }))
    for q in UNANSWERABLE:
        print(json.dumps({"question": q, "expected_source": None, "expected_pages": [], "answerable": False}))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
