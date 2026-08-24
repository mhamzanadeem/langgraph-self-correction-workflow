# 🔄 LangGraph Self-Correction Workflow

> A complete example of a **multi-step Question Answering workflow with Self-Correction** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain).

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://github.com/langchain-ai/langchain)

---

## 🎯 Overview

This project demonstrates a **self-correcting agentic workflow** that:

| Step | Description |
|------|-------------|
| 1️⃣ **Retrieve** | Fetches context from a simulated knowledge base |
| 2️⃣ **Reason** | Uses an LLM to generate an answer from context |
| 3️⃣ **Validate** | Evaluates answer quality using a second LLM pass |
| 4️⃣ **Retry** | Routes back to reasoning with feedback if invalid |
| 5️⃣ **Finalize** | Outputs the validated answer or best attempt |

The workflow **retries up to a configurable maximum** (default: 3 attempts) before finalizing.

---

## 🏗️ Architecture

```
┌─────────┐
│  query  │
└────┬────┘
     │
     ▼
┌─────────┐     ┌──────────┐
│ reason  │────▶│ validate │
└────┬────┘     └────┬─────┘
     │               │
     │    ┌──────────┴──────────┐
     │    ▼                     ▼
     │ ┌───────┐             ┌────────┐
     │ │ valid │             │ invalid│
     │ └───┬───┘             └────┬───┘
     │     │                      │
     ▼     ▼                      │
┌─────────────┐                   │
│  finalize   │◀──────────────────┘
└──────┬──────┘
       │
       ▼
      END
```

**Key LangGraph Features Demonstrated:**
- ✅ Stateful graph with `TypedDict` state
- ✅ Conditional edges (`should_continue` router)
- ✅ Cycles/loops with retry logic
- ✅ Shared state across nodes
- ✅ Dynamic decision-making based on state

---

## 📁 Project Structure

```text
langgraph-self-correction-workflow/
├── README.md
├── requirements.txt
├── .env.example
└── src/
    ├── __init__.py
    ├── main.py              # Entry point
    ├── config.py            # Configuration management
    ├── graph/
    │   ├── __init__.py
    │   ├── state.py         # GraphState TypedDict
    │   ├── nodes.py         # Node implementations
    │   └── graph_builder.py # Graph construction
    └── utils/
        ├── __init__.py
        └── llm_client.py    # LLM client wrapper
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <your-repository-url>
cd langgraph-self-correction-workflow

# Create virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example env file
cp .env.example .env
# Windows: copy .env.example .env

# Edit .env with your credentials
```

**Required environment variables:**
```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini        # Optional, default: gpt-4o-mini
MAX_ATTEMPTS=3                   # Optional, default: 3
LOG_LEVEL=INFO                   # Optional, default: INFO
```

### Running

```bash
# Run with default question
python -m src.main

# Run with custom question
python -m src.main "What is the capital of France?"
```

---

## 📝 Example Output

```bash
$ python -m src.main "What is the capital of France?"

[query] Retrieved context: Paris is the capital of France. It has a population of 2.1 million.
[reason] Attempt 1/3
[reason] Generated answer: The capital of France is Paris.
[validate] Validating answer from attempt 1
[validate] Validation passed.
[finalize] Final answer: The capital of France is Paris.

Final state:
{
    "question": "What is the capital of France?",
    "answer": "The capital of France is Paris.",
    "attempts": 1,
    "is_valid": true,
    "context": "Paris is the capital of France...",
    "feedback": "",
    "max_attempts": 3
}
```

---

## 🔁 Retry Behavior

When validation fails, the workflow **automatically retries** with feedback:

```
query
  ↓
reason attempt 1
  ↓
validate  ──invalid──▶ reason attempt 2 (with feedback)
  ↓                        ↓
validate ◀───invalid─────┘
  ↓
valid
  ↓
finalize
```

**Retry logic** (`graph/graph_builder.py:84-92`):
```python
def should_continue(state: GraphState) -> Literal["retry", "finalize"]:
    if state["is_valid"]:
        return "finalize"
    if state["attempts"] >= state["max_attempts"]:
        return "finalize"  # Max retries reached
    return "retry"
```

---

## 🧠 State Schema

```python
class GraphState(TypedDict):
    question: str          # Original user question
    context: str           # Retrieved knowledge base context
    answer: str            # Current generated answer
    attempts: int          # Current attempt number
    max_attempts: int      # Maximum allowed attempts
    is_valid: bool         # Validation result
    feedback: str          # Feedback for retry (if invalid)
```

---

## 🧩 Node Details

| Node | Purpose | Key Behavior |
|------|---------|--------------|
| `query_node` | Retrieve context | Simulated KB (replaceable with vector DB, Wikipedia, etc.) |
| `reason_node` | Generate answer | Receives question + context + previous feedback |
| `validate_node` | Evaluate answer | LLM-as-judge against context; returns `is_valid` + `feedback` |
| `finalize_node` | Output result | Prints final answer; terminates graph |

---

## 🛠️ Extending the Project

### Replace the Knowledge Base

The `query_node` uses a simulated KB. Swap it with any retriever:

```python
# graph/nodes.py - query_node
def query_node(state: GraphState) -> GraphState:
    # Replace with your retriever:
    # documents = vector_store.similarity_search(state["question"])
    # context = "\n".join([doc.page_content for doc in documents])
    
    context = SIMULATED_KB.get(state["question"], "No context found.")
    return {"context": context}
```

**Popular integrations:**
- 🔍 Vector DBs: Pinecone, Chroma, Weaviate, Qdrant
- 📚 Wikipedia, Elasticsearch, PostgreSQL/pgvector
- 🌐 Web search APIs (SerpAPI, Tavily)
- 🏢 Internal company knowledge bases

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Graph-based workflow orchestration |
| `langchain` | LLM application framework |
| `langchain-openai` | OpenAI integration |
| `python-dotenv` | Environment variable management |

---

## 🐛 Error Handling

- All LLM calls wrapped in `try/except`
- Errors logged via Python's `logging` module
- Failures recorded in state → workflow continues to validation/retry
- **Production tip**: Add tracing (LangSmith, OpenTelemetry) and custom exception handling

---

## 📚 Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/)
- [LangChain Documentation](https://python.langchain.com/)
- [Self-Correction Patterns](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)

---
