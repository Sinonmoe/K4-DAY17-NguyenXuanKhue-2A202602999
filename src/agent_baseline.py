from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Characteristics:
    - Within-session memory only (messages kept only per thread_id).
    - No persistent memory (no User.md).
    - Forgets all facts across different thread_ids.
    - No compact memory (compaction_count is always 0).
    - Accumulates token_usage and prompt_tokens_processed per turn.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live chat model if available, otherwise deterministic offline reply."""
        if not self.force_offline and self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative agent tokens generated in this thread (Agent tokens only)."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt tokens processed across all turns in this thread."""
        session = self.sessions.get(thread_id)
        if session is None:
            return 0
        return session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory mechanism and never compacts."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply with per-turn token accounting."""
        session = self.sessions.setdefault(thread_id, SessionState())
        current_user_msg = {"role": "user", "content": message}

        # Prompt context consists of all historical messages in this thread + the current message
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.messages.append(current_user_msg)

        reply_text = self._offline_response(session, message)
        reply_tokens = estimate_tokens(reply_text)

        session.messages.append({"role": "assistant", "content": reply_text})

        # Accumulate metrics immediately per turn
        session.token_usage += reply_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": reply_text,
            "content": reply_text,
            "agent_tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        """Generate deterministic response strictly based on this session's history."""
        # Prior messages before current user message (which was appended at session.messages[-1])
        prior_messages = session.messages[:-1]
        prior_text = " ".join(m["content"] for m in prior_messages)
        lower_msg = message.lower()

        is_question = any(
            q in lower_msg
            for q in [
                "nhắc lại", "tên gì", "ở đâu", "nghề gì", "đồ uống",
                "món ăn", "nuôi con gì", "style", "ai không?", "đâu mới là",
                "có biết", "tóm tắt ngắn",
            ]
        ) or lower_msg.endswith("?")

        if is_question:
            # If no history in THIS thread, baseline remembers nothing
            if not prior_text:
                return "Tôi không có thông tin về bạn trong phiên trò chuyện mới này."

            answers: list[str] = []
            if "tên" in lower_msg:
                if "dũngct stress" in prior_text.lower():
                    answers.append("Bạn tên là DũngCT Stress")
                elif "dũngct" in prior_text.lower():
                    answers.append("Bạn tên là DũngCT")

            if "nơi ở" in lower_msg or "ở đâu" in lower_msg:
                if "đà nẵng" in prior_text.lower():
                    answers.append("nơi ở: Đà Nẵng")
                elif "huế" in prior_text.lower():
                    answers.append("nơi ở: Huế")

            if "nghề" in lower_msg:
                if "mlops engineer" in prior_text.lower():
                    answers.append("nghề nghiệp: MLOps engineer")
                elif "backend engineer" in prior_text.lower():
                    answers.append("nghề nghiệp: backend engineer")

            if "đồ uống" in lower_msg and "cà phê sữa đá" in prior_text.lower():
                answers.append("đồ uống yêu thích: cà phê sữa đá")

            if "món ăn" in lower_msg and "mì quảng" in prior_text.lower():
                answers.append("món ăn yêu thích: mì Quảng")

            if "nuôi" in lower_msg and "corgi" in prior_text.lower():
                answers.append("thú cưng: corgi")

            if "style" in lower_msg:
                if "3 bullet" in prior_text.lower():
                    answers.append("style: 3 bullet")
                elif "ngắn gọn" in prior_text.lower():
                    answers.append("style: ngắn gọn")

            if answers:
                return "Dựa trên hội thoại hiện tại: " + ", ".join(answers) + "."
            return "Tôi chưa thấy thông tin đó được nhắc đến trong phiên này."

        return "Tôi đã nhận thông tin và sẽ ghi nhớ trong phiên hội thoại này."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Execute live reply with the configured chat model."""
        session = self.sessions.setdefault(thread_id, SessionState())
        current_user_msg = {"role": "user", "content": message}

        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.messages.append(current_user_msg)

        from langchain_core.messages import AIMessage, HumanMessage

        history = []
        for m in session.messages:
            if m["role"] == "user":
                history.append(HumanMessage(content=m["content"]))
            else:
                history.append(AIMessage(content=m["content"]))

        try:
            ai_msg = self.langchain_agent.invoke(history)
            reply_text = str(ai_msg.content)
        except Exception:
            reply_text = self._offline_response(session, message)

        reply_tokens = estimate_tokens(reply_text)
        session.messages.append({"role": "assistant", "content": reply_text})

        session.token_usage += reply_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": reply_text,
            "content": reply_text,
            "agent_tokens": reply_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally build chat model if API credentials are valid and live mode requested."""
        if self.force_offline:
            return None
        # In offline benchmark or test environments without real keys, keep None
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
