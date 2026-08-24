"""Build and compile the LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    finalize_node,
    query_node,
    reason_node,
    should_continue,
    validate_node,
)
from src.graph.state import GraphState


def build_graph():
    """Construct and compile the question-answering graph.

    Graph structure:

        START
          |
          v
        query
          |
          v
        reason <-------------+
          |                  |
          v                  |
       validate              |
          |                  |
       +--+--+               |
       |     |               |
     valid invalid            |
       |     |               |
       |     +---------------+
       |          retry
       v
     finalize
       |
      END
    """

    graph = StateGraph(GraphState)

    # Register graph nodes.
    graph.add_node("query", query_node)
    graph.add_node("reason", reason_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize", finalize_node)

    # Entry point.
    graph.add_edge(START, "query")

    # Normal sequential transitions.
    graph.add_edge("query", "reason")
    graph.add_edge("reason", "validate")

    # Conditional routing creates the retry loop.
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {
            "retry": "reason",
            "finalize": "finalize",
        },
    )

    # Final node terminates the graph.
    graph.add_edge("finalize", END)

    return graph.compile()