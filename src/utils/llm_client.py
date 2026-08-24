"""LLM client factory."""

from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL


def get_llm() -> ChatOpenAI:
    """Create and return the configured OpenAI chat model.

    Temperature is set to zero to make the reasoning and validation
    behavior as deterministic as practical.
    """

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Set it in the environment or .env file."
        )

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )