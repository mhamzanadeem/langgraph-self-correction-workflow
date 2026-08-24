"""State definition for the LangGraph workflow."""

from typing import TypedDict


class GraphState(TypedDict):
    """State shared by every node in the graph.

    LangGraph passes this state from node to node. Each node returns
    updates to the fields it is responsible for.

    This is one of the major differences from a simple linear chain:
    a graph maintains explicit shared state that can be inspected and
    used to make routing decisions.
    """

    question: str
    context: str
    answer: str
    attempts: int
    max_attempts: int
    is_valid: bool
    feedback: str