import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> None:
    """Load `.env`. A variable already set in the shell wins, EXCEPT when the shell exports it empty
    (e.g. `export ANTHROPIC_API_KEY=`), which python-dotenv would otherwise treat as 'already set' and ignore the file."""
    load_dotenv(path)
    for key, value in dotenv_values(path).items():
        if value and not os.environ.get(key):
            os.environ[key] = value


load_env(ROOT / ".env")

DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "agent.db"
EXT_DIR = DATA_DIR / "extensions"
SESSIONS_DIR = DATA_DIR / "sessions"
BUILTIN_SKILLS_DIR = ROOT / "skills"

# Optional feature packs, shipped with the app but switchable. Diagrams: draw.io creation/reading (tools + skill).
DIAGRAMS_ENABLED = os.getenv("OPENJANUS_DIAGRAMS", "1").strip().lower() not in ("0", "false", "no", "off")
DIAGRAMS_DIR = DATA_DIR / "diagrams"
IMAGES_ENABLED = DIAGRAMS_ENABLED  # image attachments ship with the diagrams pack for now (kept separate so they can be split later)
UPLOADS_DIR = DATA_DIR / "uploads"
OPTIONAL_SKILLS_DIRS = [ROOT / "optional" / "diagrams" / "skills"] if DIAGRAMS_ENABLED else []

DEFAULT_MODELS = {
    "claude": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "ollama": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
}
# The generic ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN variables are deliberately NOT used: shells often set them to point
# Claude Code at another backend (e.g. Ollama), which would silently redirect this app's "Claude" option. Use the
# OPENJANUS_-prefixed variable if you really want a gateway/proxy.
ANTHROPIC_BASE_URL = os.getenv("OPENJANUS_ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

MAX_STEPS = 12  # max LLM round-trips per user message
MAX_TOOL_RESULT_CHARS = 20_000
