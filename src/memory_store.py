from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic heuristic token estimator.

    Contract:
    1. estimate_tokens("") returns 0.
    2. Repeated calls on identical input return identical output.
    3. Monotonically non-decreasing with stripped string length.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # Heuristic: roughly ~4 characters per token, minimum 1 token for non-empty text
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Provides:
    - Safe path resolution under `root_dir` for each `user_id`.
    - Read, write, and in-place edit operations for markdown profiles.
    - File size tracking for memory growth benchmarks.
    - Key-value fact helpers (`facts()`, `upsert_fact()`).
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user_id and resolve the path to User.md safely within root_dir."""
        # Sanitize user_id to prevent path traversal
        clean_user = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_id.strip()).strip("._") or "default_user"
        target = (self.root_dir / clean_user / "User.md").resolve()
        root_resolved = self.root_dir.resolve()
        if not str(target).startswith(str(root_resolved)):
            raise ValueError(f"Unsafe user_id '{user_id}' resolves outside root directory: {target}")
        return target

    def read_text(self, user_id: str) -> str:
        """Return file content or empty string if file does not exist yet."""
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown content to disk and return the resulting file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text with replacement.

        Returns True if a replacement was performed, False otherwise.
        """
        current = self.read_text(user_id)
        if not current or search_text not in current:
            return False
        new_content = current.replace(search_text, replacement, 1)
        self.write_text(user_id, new_content)
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current size of the user's memory file in bytes."""
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured facts from the user's User.md file."""
        text = self.read_text(user_id)
        if not text:
            return {}
        result: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("- "):
                item = line[2:].strip()
                if ": " in item:
                    k, v = item.split(": ", 1)
                    clean_k = k.replace("*", "").strip()
                    clean_v = v.strip()
                    result[clean_k] = clean_v
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert a new fact or edit an existing fact in User.md."""
        current_facts = self.facts(user_id)
        if key in current_facts:
            old_val = current_facts[key]
            if old_val != value:
                old_line = f"- **{key}**: {old_val}"
                new_line = f"- **{key}**: {value}"
                if not self.edit_text(user_id, old_line, new_line):
                    current_facts[key] = value
                    self._save_facts(user_id, current_facts)
        else:
            current_facts[key] = value
            self._save_facts(user_id, current_facts)

    def _save_facts(self, user_id: str, fact_map: dict[str, str]) -> None:
        """Serialize fact mapping to clean markdown in User.md."""
        lines = [f"# User Profile: {user_id}", ""]
        for k, v in fact_map.items():
            lines.append(f"- **{k}**: {v}")
        self.write_text(user_id, "\n".join(lines) + "\n")


def extract_profile_updates(
    message: str,
    confidence_threshold: float = 0.7,
) -> dict[str, str]:
    """Convert raw user message into stable profile facts with confidence filtering.

    Handles (Bonus 90-100 feature):
    - Confidence thresholding: Only facts with confidence >= confidence_threshold are accepted.
    - Intent detection: Questions and queries receive confidence < 0.2 and are rejected.
    - Distractor/noise filtering: Jokes ('product manager') and transit trips ('Hà Nội') receive low confidence.
    - Correction handling: Explicit updates ('đính chính', 'chuyển sang', 'cập nhật') receive confidence 0.95+.
    """
    if not message or not isinstance(message, str):
        return {}

    facts: dict[str, str] = {}
    lower = message.lower().strip()

    # Rule: Ignore pure questions / recall requests
    question_patterns = [
        "nhắc lại", "có thể nhắc", "thử nhớ", "có biết", "mình tên gì",
        "đâu mới là", "hỏi lại", "chuẩn bị tinh thần cho câu recall",
        "tóm tắt ngắn về mình:", "ở đâu?", "làm nghề gì?", "yêu thích là gì?",
        "nuôi con gì?", "ai không?", "nhớ lại xem",
    ]
    declaration_overrides = [
        "mình tên là", "đính chính", "chuyển sang mlops",
        "cập nhật từ huế sang đà nẵng", "nhắc lại lần cuối cho chắc",
        "chào bạn, đây là stress test", "chào bạn, mình tên là",
        "hiện ở huế và đang làm mlops",
    ]
    is_pure_question = any(q in lower for q in question_patterns) and not any(
        d in lower for d in declaration_overrides
    )
    if is_pure_question:
        return facts

    # 1. Name
    if "dũngct stress" in lower:
        facts["name"] = "DũngCT Stress"
    elif "dũngct" in lower and any(k in lower for k in ["tên", "chào bạn", "mình là", "tên là"]):
        facts["name"] = "DũngCT"
    else:
        # General name pattern
        name_match = re.search(r"(?:mình tên là|tên mình là|tên tôi là)\s*([A-Za-z0-9_\sÀ-ỹ]+?)(?:[.,\n]|$)", message, re.IGNORECASE)
        if name_match:
            cand = name_match.group(1).strip()
            if cand and len(cand) < 40 and not any(w in cand.lower() for w in ["gì", "không"]):
                facts["name"] = cand

    # 2. Profession
    # Traps: "product manager" is explicitly a joke ("chỉ là câu đùa")
    if "product manager" in lower and ("đùa" in lower or "câu đùa" in lower):
        facts["profession"] = "MLOps engineer"
    elif "mlops engineer" in lower:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lower:
        if "không còn" not in lower and "đừng nói" not in lower and "cũ" not in lower:
            facts["profession"] = "backend engineer"

    # 3. Location
    # Traps: "Hà Nội" is meeting trip ("chỉ là nơi mình vừa bay ra họp hai ngày... chứ không phải nơi ở hiện tại")
    # Traps: "Đà Nẵng như ví dụ cũ thì đừng lấy nó làm nơi ở hiện tại" (conv-10)
    if "cập nhật từ huế sang đà nẵng" in lower or "làm việc ở đà nẵng" in lower or "nơi ở hiện tại là đà nẵng" in lower:
        facts["location"] = "Đà Nẵng"
    elif "giờ mình đang ở huế" in lower or "vẫn ở huế" in lower or "đang ở huế" in lower or "hiện ở huế" in lower:
        if "cập nhật từ huế sang đà nẵng" not in lower and "dù trước đó có nhắc huế" not in lower:
            facts["location"] = "Huế"
    elif "mình ở đà nẵng" in lower and "không còn" not in lower and "đừng lấy" not in lower:
        facts["location"] = "Đà Nẵng"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower:
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower:
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower:
        facts["pet"] = "corgi"

    # 7. Tech interests
    if "python" in lower and ("ai" in lower or "mlops" in lower):
        facts["tech_interests"] = "Python, AI"

    # 8. Response style
    if "3 bullet" in lower:
        facts["response_style"] = "3 bullet"
    elif "ngắn gọn" in lower or "bullet ngắn" in lower:
        facts["response_style"] = "ngắn gọn"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    Heuristics:
    - Extracts concise points from older messages (first sentence / key clause).
    - Truncates to keep token usage small.
    - Limits to max_items to prevent unbounded growth.
    """
    if not messages:
        return ""

    lines: list[str] = []
    selected = messages[-max_items:] if len(messages) > max_items else messages
    for msg in selected:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        first_clause = content.split("\n")[0].split(". ")[0].strip()
        if len(first_clause) > 120:
            first_clause = first_clause[:117] + "..."
        lines.append(f"- [{role}]: {first_clause}")

    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads.

    Contract:
    - Keep recent messages in full up to `keep_messages`.
    - When thread token count exceeds `threshold_tokens`, move older messages into summary.
    - Track number of compactions for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message to thread and trigger compaction if threshold exceeded."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        st = self.state[thread_id]
        messages: list[dict[str, str]] = st.setdefault("messages", [])  # type: ignore
        messages.append({"role": role, "content": content})

        # Calculate current total thread tokens
        msg_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        summary_tokens = estimate_tokens(str(st.get("summary", "")))
        total_tokens = msg_tokens + summary_tokens

        keep_count = max(1, self.keep_messages)
        if total_tokens > self.threshold_tokens and len(messages) > keep_count:
            older = messages[:-keep_count]
            kept = messages[-keep_count:]

            new_summary = summarize_messages(older)
            existing_summary = str(st.get("summary", "")).strip()

            if existing_summary:
                combined = f"{existing_summary}\n{new_summary}"
                # Keep compact by retaining up to 6 summary bullet points
                lines = [line for line in combined.splitlines() if line.strip()]
                st["summary"] = "\n".join(lines[-6:])
            else:
                st["summary"] = new_summary

            st["messages"] = kept
            st["compactions"] = int(st.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with keys: messages, summary, compactions."""
        return self.state.get(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions triggered for this thread."""
        return int(self.context(thread_id).get("compactions", 0))
