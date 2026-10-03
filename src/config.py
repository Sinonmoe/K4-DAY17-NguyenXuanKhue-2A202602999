from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs) -> bool:
        return False

try:
    from model_provider import ProviderConfig, normalize_provider
except ImportError:
    from src.model_provider import ProviderConfig, normalize_provider


DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "custom": "default",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    Dataclass fields:
    - base_dir: Path to the repository root.
    - data_dir: Path to the dataset directory.
    - state_dir: Path to the runtime state directory (stores User.md profiles).
    - compact_threshold_tokens: Token count threshold to trigger compaction.
    - compact_keep_messages: Number of recent messages to preserve during compaction.
    - model: Provider configuration for the primary agent.
    - judge_model: Provider configuration for evaluation / judge.
    """

    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    state_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "state")
    compact_threshold_tokens: int = 800
    compact_keep_messages: int = 4
    model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
        )
    )
    judge_model: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.0,
        )
    )

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()
        self.data_dir = Path(self.data_dir).resolve()
        self.state_dir = Path(self.state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)


def _resolve_provider_config(
    provider_raw: str,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> ProviderConfig:
    """Build ProviderConfig by reading corresponding environment variables."""
    provider = normalize_provider(provider_raw)
    chosen_model = model_name or DEFAULT_MODELS.get(provider, "gpt-4o-mini")
    api_key: str | None = None
    base_url: str | None = None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
    elif provider == "custom":
        api_key = os.getenv("CUSTOM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = os.getenv("GEMINI_BASE_URL")
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL")

    return ProviderConfig(
        provider=provider,
        model_name=chosen_model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig.

    Steps:
    1. Resolve repo root (from argument or default to project root).
    2. Load .env via python-dotenv.
    3. Ensure state/ directory exists.
    4. Populate and return LabConfig instance.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env file
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    # Directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Compact memory parameters
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Primary model configuration
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL")
    main_model_config = _resolve_provider_config(
        provider_raw=llm_provider,
        model_name=llm_model,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )

    # Judge model configuration
    judge_provider = os.getenv("JUDGE_PROVIDER", llm_provider)
    judge_model = os.getenv("JUDGE_MODEL", llm_model)
    judge_model_config = _resolve_provider_config(
        provider_raw=judge_provider,
        model_name=judge_model,
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=main_model_config,
        judge_model=judge_model_config,
    )
