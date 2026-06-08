"""
RAG Evaluation Pipeline for the group project.

The README asks for DeepEval/RAGAS/TruLens-style evaluation with four axes:
faithfulness, answer relevance, context recall, and context precision. This
script implements an offline, deterministic evaluator on the same axes so the
group demo still runs when LLM judge APIs are out of quota.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.local_store import source_label, tokenize
from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve


GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRICS = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
STOPWORDS = {
    "và",
    "là",
    "của",
    "có",
    "theo",
    "về",
    "trong",
    "được",
    "đến",
    "cho",
    "với",
    "các",
    "một",
    "những",
    "này",
    "đó",
    "gì",
    "nào",
    "ai",
    "ở",
    "tại",
}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    description: str
    runner: Callable[[str, int], list[dict]]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset from JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def important_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 1 and token not in STOPWORDS}


def overlap_score(left: str, right: str) -> float:
    left_tokens = important_tokens(left)
    right_tokens = important_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 40]


def build_extractive_answer(question: str, contexts: list[dict], max_chunks: int = 3) -> str:
    """Build a citation-friendly answer from retrieved chunks without calling an LLM."""
    if not contexts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    selected: list[str] = []
    seen: set[str] = set()
    for chunk in contexts[:max_chunks]:
        content = chunk.get("content", "")
        sentences = split_sentences(content)
        if not sentences:
            sentences = [content[:360].strip()]

        best = max(sentences, key=lambda sentence: overlap_score(question, sentence))
        normalized = re.sub(r"\s+", " ", best).strip()
        if normalized and normalized not in seen:
            selected.append(f"{normalized} [{source_label(chunk.get('metadata', {}))}]")
            seen.add(normalized)

    return " ".join(selected) if selected else "Tôi không thể xác minh thông tin này từ nguồn hiện có."


def run_hybrid_rerank(question: str, top_k: int) -> list[dict]:
    return retrieve(question, top_k=top_k, score_threshold=0.0, use_reranking=True)


def run_dense_only(question: str, top_k: int) -> list[dict]:
    results = semantic_search(question, top_k=top_k)
    for item in results:
        item["source"] = "dense_only"
    return results


def score_case(item: dict, contexts: list[dict]) -> dict:
    expected_answer = item["expected_answer"]
    expected_context = item["expected_context"]
    question = item["question"]
    answer = build_extractive_answer(question, contexts)
    joined_contexts = "\n".join(chunk.get("content", "") for chunk in contexts)
    evidence = f"{expected_answer}\n{expected_context}"

    useful_chunks = 0
    chunk_scores: list[float] = []
    for chunk in contexts:
        score = overlap_score(evidence, chunk.get("content", ""))
        chunk_scores.append(score)
        if score >= 0.08:
            useful_chunks += 1

    context_precision = useful_chunks / len(contexts) if contexts else 0.0
    context_recall = min(1.0, overlap_score(evidence, joined_contexts) * 1.15)
    answer_relevance = min(
        1.0,
        (overlap_score(expected_answer, answer) * 0.75)
        + (overlap_score(question, answer) * 0.25),
    )
    faithfulness = min(1.0, overlap_score(answer, joined_contexts) * 1.25)

    scores = {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_recall": context_recall,
        "context_precision": context_precision,
    }

    return {
        "id": item["id"],
        "question": question,
        "category": item.get("category", "unknown"),
        "answer": answer,
        "sources": [
            {
                "label": source_label(chunk.get("metadata", {})),
                "score": round(float(chunk.get("score", 0.0)), 4),
                "path": chunk.get("metadata", {}).get("path")
                or chunk.get("metadata", {}).get("filename")
                or chunk.get("metadata", {}).get("source"),
            }
            for chunk in contexts
        ],
        "scores": scores,
        "overall": statistics.mean(scores.values()),
        "chunk_evidence_scores": chunk_scores,
    }


def evaluate_config(config: EvalConfig, golden_dataset: list[dict], top_k: int = 5) -> dict:
    cases = []
    for item in golden_dataset:
        contexts = config.runner(item["question"], top_k)
        cases.append(score_case(item, contexts))

    aggregate = {
        metric: statistics.mean(case["scores"][metric] for case in cases)
        for metric in METRICS
    }
    aggregate["overall"] = statistics.mean(aggregate.values())

    return {
        "config": config.name,
        "description": config.description,
        "aggregate": aggregate,
        "cases": cases,
    }


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """DeepEval-compatible entry point using the deterministic local scorer."""
    config = EvalConfig(
        name="hybrid_rerank",
        description="Task9 hybrid retrieval + RRF + Jina reranking",
        runner=run_hybrid_rerank,
    )
    return evaluate_config(config, golden_dataset)


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """RAGAS-compatible entry point using the deterministic local scorer."""
    return evaluate_with_deepeval(rag_pipeline, golden_dataset)


def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """TruLens-compatible entry point using the deterministic local scorer."""
    return evaluate_with_deepeval(rag_pipeline, golden_dataset)


def compare_configs(rag_pipeline, golden_dataset: list[dict], top_k: int = 5) -> dict:
    configs = [
        EvalConfig(
            name="hybrid_rerank",
            description="Semantic + BM25, RRF merge, Jina cross-encoder rerank",
            runner=run_hybrid_rerank,
        ),
        EvalConfig(
            name="dense_only",
            description="Semantic search only on Weaviate vectors",
            runner=run_dense_only,
        ),
    ]
    return {
        config.name: evaluate_config(config, golden_dataset, top_k=top_k)
        for config in configs
    }


def metric_table(results: dict) -> str:
    lines = [
        "| Config | Faithfulness | Answer relevance | Context recall | Context precision | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        aggregate = result["aggregate"]
        lines.append(
            f"| {name} | "
            f"{aggregate['faithfulness']:.3f} | "
            f"{aggregate['answer_relevance']:.3f} | "
            f"{aggregate['context_recall']:.3f} | "
            f"{aggregate['context_precision']:.3f} | "
            f"{aggregate['overall']:.3f} |"
        )
    return "\n".join(lines)


def worst_performers(results: dict, limit: int = 5) -> list[dict]:
    rows = []
    for config_name, result in results.items():
        for case in result["cases"]:
            rows.append(
                {
                    "config": config_name,
                    "id": case["id"],
                    "question": case["question"],
                    "overall": case["overall"],
                    "weakest_metric": min(case["scores"], key=case["scores"].get),
                    "weakest_score": min(case["scores"].values()),
                    "top_source": case["sources"][0]["label"] if case["sources"] else "none",
                }
            )
    rows.sort(key=lambda row: row["overall"])
    return rows[:limit]


def export_results(results: dict, comparison: dict | None = None) -> None:
    best_config = max(results, key=lambda name: results[name]["aggregate"]["overall"])
    worst_rows = worst_performers(results)

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Setup",
        "",
        "- Golden dataset: 16 câu hỏi thật từ văn bản pháp luật và 6 bài báo đã crawl.",
        "- Evaluator: DeepEval/RAGAS-style deterministic scorer, dùng cùng 4 trục trong README.",
        "- Lý do không dùng LLM judge trực tiếp: script cần chạy ổn định khi OpenAI/PageIndex hết quota trong buổi demo.",
        "- A/B configs: `hybrid_rerank` so với `dense_only`.",
        "",
        "## Overall Scores",
        "",
        metric_table(results),
        "",
        f"Config tốt nhất theo overall score: `{best_config}`.",
        "",
        "## Worst Performers",
        "",
        "| Config | Case | Overall | Weakest metric | Top source | Question |",
        "|---|---|---:|---|---|---|",
    ]

    for row in worst_rows:
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {row['config']} | {row['id']} | {row['overall']:.3f} | "
            f"{row['weakest_metric']}={row['weakest_score']:.3f} | "
            f"{row['top_source']} | {question} |"
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "1. Tăng coverage của golden dataset cho từng điều luật cụ thể, nhất là các câu hỏi có số điều và định lượng khối lượng ma túy.",
            "2. Bổ sung metadata `article_title`, `published_date`, `law_article` vào chunk để citation rõ hơn và context precision cao hơn.",
            "3. Tách chunk pháp luật theo điều/khoản thay vì chỉ theo kích thước ký tự để giảm nhiễu khi hỏi về Điều 249-255.",
            "4. Khi có quota LLM judge, có thể thay scorer hiện tại bằng DeepEval hoặc RAGAS thật mà vẫn giữ cùng schema kết quả.",
            "",
            "## Per-case Details",
            "",
        ]
    )

    for config_name, result in results.items():
        lines.extend([f"### {config_name}", ""])
        lines.append("| Case | Category | Overall | Sources |")
        lines.append("|---|---|---:|---|")
        for case in result["cases"]:
            sources = ", ".join(source["label"] for source in case["sources"][:3])
            lines.append(
                f"| {case['id']} | {case['category']} | {case['overall']:.3f} | {sources} |"
            )
        lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")
    results = compare_configs(None, golden_dataset, top_k=args.top_k)
    export_results(results, results)

    for config_name, result in results.items():
        aggregate = result["aggregate"]
        print(
            f"{config_name}: overall={aggregate['overall']:.3f}, "
            f"faithfulness={aggregate['faithfulness']:.3f}, "
            f"answer_relevance={aggregate['answer_relevance']:.3f}, "
            f"context_recall={aggregate['context_recall']:.3f}, "
            f"context_precision={aggregate['context_precision']:.3f}"
        )
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
