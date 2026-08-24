"""Application entry point."""

import logging
import sys
from pprint import pprint

from src.config import MAX_ATTEMPTS, LOG_LEVEL, validate_config
from src.graph.graph_builder import build_graph


def configure_logging() -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Build and execute the LangGraph workflow."""

    configure_logging()

    try:
        validate_config()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    # Allow the user to provide a question on the command line.
    question = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else "What is the capital of France?"
    )

    print("=" * 70)
    print("LangGraph Question Answering with Self-Correction")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Maximum attempts: {MAX_ATTEMPTS}")
    print()

    workflow = build_graph()

    initial_state = {
        "question": question,
        "context": "",
        "answer": "",
        "attempts": 0,
        "max_attempts": MAX_ATTEMPTS,
        "is_valid": False,
        "feedback": "",
    }

    try:
        final_state = workflow.invoke(initial_state)
    except Exception as exc:
        logging.exception("Graph execution failed.")
        print(f"\nGraph execution failed: {exc}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Final State")
    print("=" * 70)
    pprint(final_state)


if __name__ == "__main__":
    main()