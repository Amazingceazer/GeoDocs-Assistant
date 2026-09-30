"""Compare retrieval modes on a labelled Q&A set.

Usage: python -m eval.run_eval [--file eval/qa_pairs.jsonl] [--k 5] [--no-judge]
Writes eval/results.md (paste the table into your README).
"""
import argparse
import json
import time
from pathlib import Path

from app.llm import provider
from app.llm.prompts import SYSTEM, build_user_prompt
from app.rag import retrieve

MODES = ["dense", "hybrid", "hybrid_rerank"]
NO_ANSWER = "I don't know based on the provided documents."

JUDGE_SYSTEM = (
    "You check whether an answer is fully supported by the context. "
    "Reply with exactly SUPPORTED if every claim in the answer is backed by the context, "
    "otherwise reply UNSUPPORTED."
)


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith("i don't know")


def is_hit(hit: dict, row: dict) -> bool:
    return hit["source"] == row["expected_source"] and hit["page"] in row["expected_pages"]


def faithful(hits: list[dict], answer: str) -> bool:
    context = "\n\n".join(h["text"] for h in hits)
    verdict = provider.generate(JUDGE_SYSTEM, f"Context:\n{context}\n\nAnswer:\n{answer}")
    return verdict.strip().upper().startswith("SUPPORTED")


def avg(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def evaluate(rows: list[dict], mode: str, k: int, judge: bool) -> dict:
    hits_at_k, rr, refused_ok, false_refusals, faith, latencies = [], [], [], [], [], []
    for row in rows:
        t0 = time.perf_counter()
        hits = retrieve(row["question"], k, mode)
        latencies.append(time.perf_counter() - t0)

        answer = NO_ANSWER
        if hits:
            answer = provider.generate(SYSTEM, build_user_prompt(row["question"], hits))
        refused = is_refusal(answer)

        if not row["answerable"]:
            refused_ok.append(refused)
            continue

        rank = next((i for i, h in enumerate(hits, 1) if is_hit(h, row)), None)
        hits_at_k.append(rank is not None)
        rr.append(1 / rank if rank else 0.0)
        false_refusals.append(refused)
        if judge and not refused:
            faith.append(faithful(hits, answer))

    return {
        "mode": mode,
        f"hit@{k}": avg(hits_at_k),
        "MRR": avg(rr),
        "faithfulness": avg(faith),
        "correct refusals": avg(refused_ok),
        "false refusals": avg(false_refusals),
        "retrieval ms": avg(latencies) * 1000,
    }


def fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def to_markdown(results: list[dict]) -> str:
    cols = list(results[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in results]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/qa_pairs.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.file).read_text().splitlines() if line.strip()]
    results = [evaluate(rows, m, args.k, not args.no_judge) for m in MODES]
    table = to_markdown(results)
    print(table)
    Path("eval/results.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
