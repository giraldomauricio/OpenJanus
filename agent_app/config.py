import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "agent.db"
EXT_DIR = DATA_DIR / "extensions"
SESSIONS_DIR = DATA_DIR / "sessions"
BUILTIN_SKILLS_DIR = ROOT / "skills"

DEFAULT_MODELS = {
    "claude": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "ollama": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
}
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

MAX_STEPS = 12  # max LLM round-trips per user message
MAX_TOOL_RESULT_CHARS = 20_000
