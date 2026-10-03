from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root)
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")
    user_id = "test_user"

    # 1. Read non-existent
    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    # 2. Write initial markdown
    initial_content = "# User Profile: test_user\n\n- **location**: Đà Nẵng\n- **profession**: backend engineer\n"
    path = store.write_text(user_id, initial_content)
    assert path.exists()
    assert store.file_size(user_id) > 0
    assert store.read_text(user_id) == initial_content

    # 3. Edit in-place
    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    updated = store.read_text(user_id)
    assert "Huế" in updated
    assert "Đà Nẵng" not in updated

    # 4. Edit non-existent text
    assert store.edit_text(user_id, "Hà Nội", "Sài Gòn") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    manager = CompactMemoryManager(
        threshold_tokens=config.compact_threshold_tokens,
        keep_messages=config.compact_keep_messages,
    )
    thread_id = "test_thread"

    # 1. Message under threshold
    manager.append(thread_id, "user", "Xin chào!")
    assert manager.compaction_count(thread_id) == 0
    assert len(manager.context(thread_id)["messages"]) == 1

    # 2. Append long messages exceeding threshold
    for i in range(8):
        manager.append(
            thread_id,
            "user",
            f"Đây là tin tức dài số {i} với rất nhiều nội dung kỹ thuật để vượt ngưỡng 60 token và kích hoạt compaction.",
        )

    # 3. Assert compaction triggered
    assert manager.compaction_count(thread_id) > 0
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) <= config.compact_keep_messages
    assert len(str(ctx["summary"])) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)
    user_id = "test_dungct"

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    fact_msg = "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và làm backend engineer."
    recall_msg = "Mình tên gì?"

    # Thread 1: Both receive the facts
    baseline.reply(user_id, "thread_1", fact_msg)
    advanced.reply(user_id, "thread_1", fact_msg)

    # Thread 2: Both are asked in a new thread
    base_reply = baseline.reply(user_id, "thread_2", recall_msg)["response"]
    adv_reply = advanced.reply(user_id, "thread_2", recall_msg)["response"]

    # Advanced MUST remember across sessions
    assert "DũngCT" in adv_reply, f"Advanced should remember across sessions: {adv_reply}"

    # Baseline MUST NOT remember across sessions
    assert "DũngCT" not in base_reply, f"Baseline should forget across sessions: {base_reply}"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    thread_id = "long_thread"
    user_id = "stress_user"

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    long_turns = [
        f"Lượt {i}: Đây là đoạn tin tức công nghệ dài về hệ thống Artemis III và X-59 với rất nhiều chi tiết kỹ thuật "
        f"nhằm làm cho ngữ cảnh tích lũy của hội thoại ngày càng nặng nề qua từng bước kiểm thử."
        for i in range(12)
    ]

    for turn in long_turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)

    # Advanced must trigger compaction
    assert advanced.compaction_count(thread_id) > 0
    # Advanced must carry fewer prompt tokens than Baseline
    assert advanced_prompt_tokens < baseline_prompt_tokens, (
        f"Advanced ({advanced_prompt_tokens}) should process fewer prompt tokens than Baseline ({baseline_prompt_tokens})"
    )
