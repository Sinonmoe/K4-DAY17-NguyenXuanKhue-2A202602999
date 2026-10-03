from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0, 0.5, or 1 depending on how many expected facts appear.

    Three-level grading:
    - 1.0: all expected facts are present
    - 0.5: at least one fact present, but not all (partially correct)
    - 0.0: none of the expected facts are present
    """
    if not expected:
        return 1.0
    hits = sum(1 for exp in expected if exp.lower() in answer.lower())
    if hits == len(expected):
        return 1.0
    elif hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode (0.0 - 1.0).

    Components:
    - Fact recall score (weight 0.7)
    - Structural fluency / non-empty answer (weight 0.3)
    """
    if not answer or not answer.strip():
        return 0.0

    rec = recall_points(answer, expected)
    length = len(answer.strip())
    structure_score = 1.0 if 15 <= length <= 600 else 0.5
    return round(0.7 * rec + 0.3 * structure_score, 2)


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: Any,
) -> BenchmarkRow:
    """Evaluate one agent over a conversation suite.

    Procedure:
    1. Feed all turns to the agent in dedicated chat threads.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in fresh threads.
    5. Compute average recall and response quality.
    6. Record memory file growth and compaction count.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []

    unique_users = {c["user_id"] for c in conversations}
    initial_mem_bytes = 0
    if hasattr(agent, "memory_file_size"):
        initial_mem_bytes = sum(agent.memory_file_size(u) for u in unique_users)

    for c in conversations:
        conv_id = c["id"]
        user_id = c["user_id"]
        chat_thread_id = f"{conv_id}_chat"

        # 1. Feed all turns to the agent
        for turn in c.get("turns", []):
            agent.reply(user_id=user_id, thread_id=chat_thread_id, message=turn)

        # 2. Track metrics from chat turns
        total_agent_tokens += agent.token_usage(chat_thread_id)
        total_prompt_tokens += agent.prompt_token_usage(chat_thread_id)
        total_compactions += agent.compaction_count(chat_thread_id)

        # 3. Ask recall questions in a FRESH thread per conversation
        recall_thread_id = f"{conv_id}_recall"
        for q in c.get("recall_questions", []):
            question_text = q["question"]
            expected = q.get("expected_contains", [])

            res = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=question_text)
            answer = res.get("response", res.get("content", ""))

            rec_score = recall_points(answer, expected)
            qual_score = heuristic_quality(answer, expected)
            all_recall_scores.append(rec_score)
            all_quality_scores.append(qual_score)

    avg_recall = (sum(all_recall_scores) / len(all_recall_scores)) if all_recall_scores else 0.0
    avg_quality = (sum(all_quality_scores) / len(all_quality_scores)) if all_quality_scores else 0.0

    final_mem_bytes = 0
    if hasattr(agent, "memory_file_size"):
        final_mem_bytes = sum(agent.memory_file_size(u) for u in unique_users)
    memory_growth = max(0, final_mem_bytes - initial_mem_bytes)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows into a markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github", disable_numparse=True)
    except ImportError:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in data]
        return "\n".join([header_line, sep_line] + body_lines)


def main() -> None:
    """Run both benchmark suites and display the comparison tables."""
    root = Path(__file__).resolve().parent.parent
    config = load_config(root)

    std_convs = load_conversations(config.data_dir / "conversations.json")
    stress_convs = load_conversations(config.data_dir / "advanced_long_context.json")

    # 1. Standard Benchmark
    print("\n" + "=" * 90)
    print("### Standard Benchmark (`data/conversations.json`)")
    print("=" * 90)

    shutil.rmtree(config.state_dir / "profiles" / "dungct", ignore_errors=True)

    base_std = BaselineAgent(config=config, force_offline=True)
    row_base_std = run_agent_benchmark("Baseline", base_std, std_convs, config)

    adv_std = AdvancedAgent(config=config, force_offline=True)
    row_adv_std = run_agent_benchmark("Advanced", adv_std, std_convs, config)

    print(format_rows([row_base_std, row_adv_std]))

    # 2. Long-Context Stress Benchmark
    print("\n" + "=" * 90)
    print("### Long-Context Stress Benchmark (`data/advanced_long_context.json`)")
    print("=" * 90)

    shutil.rmtree(config.state_dir / "profiles" / "dungct_stress", ignore_errors=True)

    base_stress = BaselineAgent(config=config, force_offline=True)
    row_base_stress = run_agent_benchmark("Baseline", base_stress, stress_convs, config)

    adv_stress = AdvancedAgent(config=config, force_offline=True)
    row_adv_stress = run_agent_benchmark("Advanced", adv_stress, stress_convs, config)

    print(format_rows([row_base_stress, row_adv_stress]))
    print()


if __name__ == "__main__":
    main()
