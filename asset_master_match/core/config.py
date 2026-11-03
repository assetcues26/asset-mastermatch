"""Environment-driven configuration. Secrets live in local.settings.json / app settings."""

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y")


# Gemini
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
GEMINI_CALL_TIMEOUT_SEC = _env_float("GEMINI_CALL_TIMEOUT_SEC", 30.0)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 3)
GEMINI_MAX_INPUT_TOKENS = _env_int("GEMINI_MAX_INPUT_TOKENS", 8000)
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 1500)
GEMINI_DEFAULT_OUTPUT_TOKENS = _env_int("GEMINI_DEFAULT_OUTPUT_TOKENS", 800)
GEMINI_MAX_CALLS_PER_MINUTE = _env_int("GEMINI_MAX_CALLS_PER_MINUTE", 120)
GEMINI_MAX_CALLS_PER_HOUR = _env_int("GEMINI_MAX_CALLS_PER_HOUR", 500)
GEMINI_MAX_CALLS_PER_DAY = _env_int("GEMINI_MAX_CALLS_PER_DAY", 5000)

# Prompt char budget (~4 chars/token), overridable via GEMINI_MAX_PROMPT_CHARS
MAX_PROMPT_LENGTH = _env_int(
    "GEMINI_MAX_PROMPT_CHARS",
    GEMINI_MAX_INPUT_TOKENS * 4,
)

# Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 1536)

# Pinecone
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "")
PINECONE_TOP_K = _env_int("PINECONE_TOP_K", 5)
PINECONE_MAX_CALLS_PER_MINUTE = _env_int("PINECONE_MAX_CALLS_PER_MINUTE", 100)
PINECONE_MAX_CALLS_PER_HOUR = _env_int("PINECONE_MAX_CALLS_PER_HOUR", 1000)
PINECONE_MAX_CALLS_PER_DAY = _env_int("PINECONE_MAX_CALLS_PER_DAY", 10000)

# Request / field limits
MAX_ASSETS_PER_REQUEST = _env_int("MAX_ASSETS_PER_REQUEST", 50)
MAX_PARALLEL_ITEMS = _env_int("MAX_PARALLEL_ITEMS", 10)
MAX_ASSET_NAME_LENGTH = _env_int("MAX_ASSET_NAME_LENGTH", 800)
MAX_ASSET_DESCRIPTION_LENGTH = _env_int("MAX_ASSET_DESCRIPTION_LENGTH", 800)
MAX_EMBED_QUERY_LENGTH = _env_int("MAX_EMBED_QUERY_LENGTH", 1000)
MAX_MAKE_MODEL_LENGTH = _env_int("MAX_MAKE_MODEL_LENGTH", 150)
MAX_PROMPT_FIELD_LENGTH = _env_int("MAX_PROMPT_FIELD_LENGTH", 200)

# Prompts
PROMPT_HOT_RELOAD = _env_bool("PROMPT_HOT_RELOAD", False)


def clamp_output_tokens(requested: int | None = None) -> int:
    value = requested if requested is not None else GEMINI_DEFAULT_OUTPUT_TOKENS
    return max(64, min(int(value), GEMINI_MAX_OUTPUT_TOKENS))
