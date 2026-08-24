"""Node implementations for the LangGraph workflow."""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from src.graph.state import GraphState
from src.utils.llm_client import get_llm


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulated knowledge base
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "france": (
        "Paris is the capital of France. "
        "It has a population of approximately 2.1 million people "
        "within the city proper."
    ),
    "python": (
        "Python is a high-level, general-purpose programming language. "
        "It was created by Guido van Rossum and first released in 1991."
    ),
    "langgraph": (
        "LangGraph is a framework for building stateful, multi-step "
        "applications and agent workflows using graph-based execution."
    ),
}


# ---------------------------------------------------------------------------
# Structured validation output
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Expected structured output from the validation LLM."""

    is_valid: bool = Field(
        description="Whether the answer is correct, complete, and supported "
        "by the supplied context."
    )

    feedback: str = Field(
        description=(
            "Specific feedback for improving the answer. "
            "Use an empty string when the answer is valid."
        )
    )


# ---------------------------------------------------------------------------
# Query node
# ---------------------------------------------------------------------------

def query_node(state: GraphState) -> dict:
    """Retrieve relevant context for the user's question.

    This implementation deliberately uses a tiny simulated knowledge base.
    A production implementation could replace this function with a vector
    store, search engine, Wikipedia API, SQL query, or another retriever.
    """

    question = state["question"].lower()

    context = (
        "No directly matching document was found in the simulated "
        "knowledge base."
    )

    for keyword, document in KNOWLEDGE_BASE.items():
        if keyword in question:
            context = document
            break

    logger.info("[query] Retrieved context: %s", context)

    return {
        "context": context,
    }


# ---------------------------------------------------------------------------
# Reasoning node
# ---------------------------------------------------------------------------

def reason_node(state: GraphState) -> dict:
    """Generate an answer from the question and retrieved context.

    If validation previously failed, its feedback is included in the prompt
    so that the LLM can attempt a better answer.

    The attempts counter is incremented here, because each execution of this
    node represents one actual reasoning attempt.
    """

    next_attempt = state["attempts"] + 1

    logger.info(
        "[reason] Attempt %d/%d",
        next_attempt,
        state["max_attempts"],
    )

    feedback = state.get("feedback", "").strip()

    feedback_section = (
        f"""
Previous validation feedback:
{feedback}

Use this feedback to improve the new answer.
"""
        if feedback
        else ""
    )

    prompt = f"""
You are a careful question-answering assistant.

Answer the user's question using ONLY the supplied context.

User question:
{state["question"]}

Retrieved context:
{state["context"]}
{feedback_section}

Requirements:
1. Answer the question directly.
2. Do not invent facts that are absent from the context.
3. If the context does not contain enough information, say so.
4. Keep the answer concise.
5. Correct any issue identified by previous validation feedback.

Return only the answer text.
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)

        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()

        logger.info("[reason] Generated answer: %s", answer)

        return {
            "answer": answer,
            "attempts": next_attempt,
        }

    except Exception as exc:
        logger.exception("[reason] LLM call failed: %s", exc)

        # Returning a failed answer instead of crashing allows the validator
        # and retry mechanism to decide what happens next.
        return {
            "answer": "",
            "attempts": next_attempt,
            "feedback": f"Reasoning failed because of an LLM error: {exc}",
        }


# ---------------------------------------------------------------------------
# Validation node
# ---------------------------------------------------------------------------

def validate_node(state: GraphState) -> dict:
    """Validate the generated answer using an LLM evaluator.

    The validator receives the same context used for reasoning. This reduces
    the chance that validation approves an answer based on unsupported facts.
    """

    logger.info(
        "[validate] Validating answer from attempt %d",
        state["attempts"],
    )

    prompt = f"""
You are an answer validator.

Determine whether the proposed answer correctly answers the question
using the supplied context.

Question:
{state["question"]}

Context:
{state["context"]}

Proposed answer:
{state["answer"]}

Validation rules:
1. The answer must address the question.
2. Important factual claims must be supported by the context.
3. The answer must not hallucinate unsupported information.
4. The answer should be reasonably complete for the question.
5. If the answer is empty or clearly unusable, mark it invalid.
6. If invalid, provide concise, actionable feedback for another reasoning
   attempt.
"""

    try:
        llm = get_llm()

        # Structured output makes the validator return a predictable schema
        # instead of requiring us to manually parse arbitrary JSON.
        validator = llm.with_structured_output(ValidationResult)
        result = validator.invoke(prompt)

        logger.info(
            "[validate] is_valid=%s feedback=%s",
            result.is_valid,
            result.feedback,
        )

        if result.is_valid:
            logger.info("[validate] Validation passed.")
        else:
            logger.info("[validate] Validation failed; retrying if possible.")

        return {
            "is_valid": result.is_valid,
            "feedback": result.feedback.strip(),
        }

    except Exception as exc:
        logger.exception("[validate] LLM call failed: %s", exc)

        # Treat validator failures as invalid so the graph can retry.
        return {
            "is_valid": False,
            "feedback": (
                "Validation failed because of an LLM error. "
                "Retry the reasoning and validation process."
            ),
        }


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def should_continue(
    state: GraphState,
) -> Literal["retry", "finalize"]:
    """Determine what node should execute after validation.

    Routing rules:

    1. A valid answer goes directly to finalization.
    2. An invalid answer is retried while attempts remain.
    3. Once max_attempts is reached, force finalization even if the answer
       remains invalid.

    This conditional routing plus the edge back to `reason` creates the
    graph's retry loop.

    In a traditional LangChain chain, the workflow is normally modeled as a
    mostly fixed sequence of operations. LangGraph instead allows a node to
    inspect state and dynamically select the next node, including routing
    back to an earlier node.
    """

    if state["is_valid"]:
        return "finalize"

    if state["attempts"] >= state["max_attempts"]:
        logger.warning(
            "[router] Maximum attempts reached (%d). "
            "Forcing finalization.",
            state["max_attempts"],
        )
        return "finalize"

    return "retry"


# ---------------------------------------------------------------------------
# Finalize node
# ---------------------------------------------------------------------------

def finalize_node(state: GraphState) -> dict:
    """Finalize and display the answer.

    This node intentionally does not make another LLM call.
    """

    logger.info("[finalize] Final answer: %s", state["answer"])

    print(f'\n[finalize] Final answer: "{state["answer"]}"')

    if not state["is_valid"]:
        print(
            "[finalize] Note: the maximum number of attempts was reached "
            "without successful validation."
        )

    return {}