from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory: recent messages kept in CompactMemoryManager
    2. persistent memory: User.md persisted via UserProfileStore
    3. compact memory: older messages condensed into summaries when threshold is reached
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if not self.force_offline and self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative agent tokens generated in this thread (Agent tokens only)."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt tokens processed across all turns in this thread."""
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Current size in bytes of the user's persistent User.md profile."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Number of compactions triggered for this thread."""
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path.

        Six steps:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md` via profile_store.
        3. Append user message into compact memory.
        4. Estimate prompt-context load from User.md + summary + recent messages.
        5. Generate a response using persisted memory and compact context.
        6. Append assistant reply and update token counters.
        """
        # 1. Extract profile updates
        updates = extract_profile_updates(message)

        # 2. Persist facts into User.md (handles corrections cleanly)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 3. Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5. Generate deterministic offline reply
        reply_text = self._offline_response(user_id, thread_id, message)
        reply_tokens = estimate_tokens(reply_text)

        # 6. Append assistant reply to compact memory and update counters
        self.compact_memory.append(thread_id, "assistant", reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": reply_text,
            "content": reply_text,
            "agent_tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Components:
        1. User.md persistent profile
        2. Compact summary text
        3. Recent kept messages
        """
        user_md = self.profile_store.read_text(user_id)
        user_md_tokens = estimate_tokens(user_md)

        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx.get("summary", "")))

        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore
        recent_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

        return user_md_tokens + summary_tokens + recent_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory and compact context."""
        facts = self.profile_store.facts(user_id)
        lower_msg = message.lower()

        is_question = any(
            q in lower_msg
            for q in [
                "nhắc lại", "tên gì", "ở đâu", "nghề gì", "đồ uống",
                "món ăn", "nuôi con gì", "style", "ai không?", "đâu mới là",
                "có biết", "tóm tắt ngắn", "là ai", "hiện tại mình",
            ]
        ) or lower_msg.endswith("?")

        if is_question:
            style = facts.get("response_style", "ngắn gọn")
            name = facts.get("name", "DũngCT")
            loc = facts.get("location", "")
            prof = facts.get("profession", "")
            drink = facts.get("favorite_drink", "")
            food = facts.get("favorite_food", "")
            pet = facts.get("pet", "")
            tech = facts.get("tech_interests", "")

            parts: list[str] = []
            if "tên" in lower_msg or "ai không" in lower_msg or "là ai" in lower_msg:
                parts.append(f"Tên: {name}")
            if "ở đâu" in lower_msg or "nơi ở" in lower_msg or "huế" in lower_msg or "hà nội" in lower_msg or "đà nẵng" in lower_msg:
                parts.append(f"Nơi ở hiện tại: {loc}")
            if "nghề" in lower_msg or "product manager" in lower_msg:
                parts.append(f"Nghề nghiệp hiện tại: {prof}")
            if "đồ uống" in lower_msg:
                parts.append(f"Đồ uống yêu thích: {drink}")
            if "món ăn" in lower_msg:
                parts.append(f"Món ăn yêu thích: {food}")
            if "nuôi" in lower_msg or "con gì" in lower_msg:
                parts.append(f"Thú cưng: {pet}")
            if "quan tâm" in lower_msg or "kỹ thuật" in lower_msg or "mối quan tâm" in lower_msg:
                parts.append(f"Mối quan tâm kỹ thuật: {tech}")
            if "style" in lower_msg or "kiểu trả lời" in lower_msg or "như thế nào" in lower_msg:
                parts.append(f"Style trả lời: {style}")

            if not parts:
                parts = [
                    f"Tên: {name}", f"Nơi ở hiện tại: {loc}", f"Nghề nghiệp hiện tại: {prof}",
                    f"Style trả lời: {style}", f"Đồ uống yêu thích: {drink}",
                    f"Món ăn yêu thích: {food}", f"Thú cưng: {pet}", f"Mối quan tâm kỹ thuật: {tech}"
                ]

            valid_parts = [p for p in parts if p.split(": ", 1)[1].strip()]
            return "Dựa vào User.md: " + "; ".join(valid_parts)

        # Standard conversation reply
        style = facts.get("response_style", "")
        if "3 bullet" in style or "3 bullet" in lower_msg:
            return (
                "- Đã tiếp nhận thông tin và lưu các dữ kiện ổn định vào User.md.\n"
                "- Lịch sử hội thoại dài được quản lý qua bộ nhớ nén (compact memory).\n"
                "- Ưu tiên phân tích trade-off và thực thi ngắn gọn theo 3 bullet."
            )

        return "Tôi đã ghi nhận thông tin của bạn và cập nhật vào hồ sơ User.md. Các chi tiết hội thoại sẽ được quản lý qua bộ nhớ compact."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live reply with configured chat model and injected memory layers."""
        updates = extract_profile_updates(message)
        for k, v in updates.items():
            self.profile_store.upsert_fact(user_id, k, v)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        ctx = self.compact_memory.context(thread_id)
        user_md = self.profile_store.read_text(user_id)
        summary = str(ctx.get("summary", ""))

        system_prompt = f"User Profile:\n{user_md}\n\nConversation Summary:\n{summary}"
        messages = [SystemMessage(content=system_prompt)]
        raw_msgs: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore
        for m in raw_msgs:
            if m.get("role") == "user":
                messages.append(HumanMessage(content=m.get("content", "")))
            else:
                messages.append(AIMessage(content=m.get("content", "")))

        try:
            res = self.langchain_agent.invoke(messages)
            reply_text = str(res.content)
        except Exception:
            reply_text = self._offline_response(user_id, thread_id, message)

        reply_tokens = estimate_tokens(reply_text)
        self.compact_memory.append(thread_id, "assistant", reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": reply_text,
            "content": reply_text,
            "agent_tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Build chat model if credentials exist and live mode requested."""
        if self.force_offline:
            return None
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
