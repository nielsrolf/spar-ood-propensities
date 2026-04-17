"""Central config. Edit here to swap providers/models/thresholds."""
from __future__ import annotations
import os
from pathlib import Path

# Paths (all inside claudes/)
CLAUDES_DIR = Path(__file__).resolve().parent.parent
INPUT_SPEC = CLAUDES_DIR / "list_some3.json"
PIPELINE_DIR = CLAUDES_DIR / "pipeline"
CACHE_DIR = CLAUDES_DIR / "cache"
CONVERSATIONS_DIR = CLAUDES_DIR / "conversations"
EVALS_DIR = CLAUDES_DIR / "evals"
REPORT_PATH = CLAUDES_DIR / "report.md"
DOTENV_PATH = CLAUDES_DIR.parent.parent / ".env"

# Providers: "openai" or "anthropic"
GENERATOR_PROVIDER = "openai"
GENERATOR_MODEL = "gpt-4o-mini"   # start cheap
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = "gpt-4o-mini"

# Fallback/upgrade models if cheap ones underperform
GENERATOR_MODEL_STRONG = "gpt-4.1"
JUDGE_MODEL_STRONG = "gpt-4.1-mini"

# Concurrency
MAX_CONCURRENT = 10

# Orthogonality
TARGET_AXIS_MIN_MAGNITUDE = 55   # |target-axis score| must be >= this for a "strong" signal; level-0 uses <=15
TARGET_LEVEL0_MAX_MAGNITUDE = 15
ORTHOGONALITY_THRESHOLD = 20     # |other-axis score| must be <= this (calibrated on pilot; default ±20, updated after pilot)
MAX_REGEN_ATTEMPTS = 3
CONVERSATIONS_PER_LEVEL = 50
PILOT_CONVERSATIONS_PER_LEVEL = 8
PILOT_AXES = ["criticism", "user trust"]  # two axes for pilot

# Fixed domain list for combinatorial coverage
DOMAINS = [
    "coding",
    "writing",
    "life advice",
    "business",
    "research",
    "creative",
    "technical decision",
    "health information",
    "career",
    "education",
]

# Env loader (no external dotenv dep)
def load_env() -> None:
    if not DOTENV_PATH.exists():
        return
    for line in DOTENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)
