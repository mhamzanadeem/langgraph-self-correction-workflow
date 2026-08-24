"""Application configuration.

Environment variables are loaded from a .env file when present.
"""

import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
except ValueError:
    MAX_ATTEMPTS = 3

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def validate_config() -> None:
    """Validate required application configuration."""

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Copy .env.example to .env and add your OpenAI API key."
        )

    if MAX_ATTEMPTS < 1:
        raise RuntimeError("MAX_ATTEMPTS must be at least 1.")