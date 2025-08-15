"""
Goal:
1) Tiny LangGraph: draft -> review -> END 
2) First run: LLM drafts, LLM reviews -> usually NEEDS_REVISION -> END
3) User chooses checkpoint index to resume from (prompted in console) 
4) Resume from chosen checkpoint: LLM drafts improved text -> LLM reviews -> APPROVED

"""
# pip install -U langgraph langgraph-checkpoint-sqlite
# pip install langchain-openai
# pip install python-dotenv
# update .env file with appropriate key related to the LLM used.

# =========================
# 0) Imports (only what we need)
# =========================
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver  # ensures real checkpoint IDs
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# 1) Define the State (the data passed between nodes)
# =========================
class ChatState(TypedDict):
    """
    The conversation "state" that moves through the graph.

    A list of dict messages with 'role' and 'content'.
    """
    messages: List[Dict[str, str]]  # e.g., {"role": "user" | "assistant" | "system", "content": "..."}


# =========================
# 2) LLM helper
# =========================
def get_llm():
    """
    Returns a chat LLM client. Assuming OPENAI_API_KEY is set in the environment.
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# =========================
# 3) Node functions (the work done at each step)
# =========================
def node_create_draft(state: ChatState) -> ChatState:
    """
    Ask the LLM to write a short, beginner-friendly draft reply to the most recent user message.
    If a previous review requested revision, the system prompt nudges improving clarity.
    """
    user_last = next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "(no user message)"
    )

    # Check if reviewer previously asked for revision
    revision_requested = any(
        m["role"] == "system" and m.get("content") == "REVIEW: NEEDS_REVISION"
        for m in state["messages"]
    )

    llm = get_llm()
    system = SystemMessage(
        content=(
            "You are a concise, beginner-friendly assistant. "
            "Write a helpful draft answer to the user's message. "
            "If a revision was requested previously, improve clarity and include a brief reason using 'because'. "
            "Output one short paragraph."
        )
    )
    user = HumanMessage(
        content=(
            f"User asked: {user_last}\n"
            f"Context: {'A revision was requested earlier.' if revision_requested else 'This is the first draft.'}\n"
            f"Constraints: Plain language. One short paragraph. Include 'because' if it helps clarity."
        )
    )
    resp = llm.invoke([system, user])
    text = resp.content.strip()

    state["messages"].append({"role": "assistant", "content": f"[DRAFT] {text}"})
    return state


def node_review_draft(state: ChatState) -> ChatState:
    """
    Ask the LLM to strictly review the latest draft.
    It must output EXACTLY one of:
      DECISION: APPROVED
      DECISION: NEEDS_REVISION
    plus a one-line reason on the next line (we only parse the decision line).
    """
    last_draft = None
    for m in reversed(state["messages"]):
        if m["role"] == "assistant" and isinstance(m.get("content"), str) and m["content"].startswith("[DRAFT]"):
            last_draft = m["content"].replace("[DRAFT]", "", 1).strip()
            break

    if last_draft is None:
        state["messages"].append({"role": "system", "content": "REVIEW: NEEDS_REVISION"})
        return state

    llm = get_llm()
    system = SystemMessage(
        content=(
            "You are a strict reviewer. Decide if the draft is acceptable for a beginner.\n"
            "Output MUST be exactly:\n"
            "DECISION: APPROVED   or   DECISION: NEEDS_REVISION\n"
            "REASON: <one short line>\n"
            "Approve if the draft is clear, correct, and includes a brief explanation (e.g., uses 'because') or is otherwise sufficient.\n"
            "Otherwise, request revision."
        )
    )
    user = HumanMessage(
        content=f"Draft to review:\n---\n{last_draft}\n---"
    )
    verdict = llm.invoke([system, user]).content.strip().upper().splitlines()[0]
    decision = "REVIEW: NEEDS_REVISION"
    if "DECISION:" in verdict:
        if "APPROVED" in verdict:
            decision = "REVIEW: APPROVED"
        elif "NEEDS_REVISION" in verdict:
            decision = "REVIEW: NEEDS_REVISION"

    state["messages"].append({"role": "system", "content": decision})
    return state

# =========================
# 4) Build the graph (wire nodes + routing)
# =========================
def build_graph():
    """
    Create and return a compiled LangGraph application with a SQLite checkpointer.
    Flow:
      entry -> node_create_draft -> node_review_draft -> END (HITL stop after review)
    """
    g = StateGraph(ChatState)
    g.add_node("draft", node_create_draft)
    g.add_node("review", node_review_draft)
    g.set_entry_point("draft")

    # After draft: go to review (we don't end here)
    def after_draft(_: ChatState):
        return "review"

    # After review: HITL-style — always stop. You'll resume later if needed.
    def after_review(_: ChatState):
        return END

    g.add_conditional_edges("draft", after_draft)
    g.add_conditional_edges("review", after_review)

    # SQLite checkpointer so checkpoint IDs are persisted + exposed
    conn = sqlite3.connect("lg_checkpoints.db", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # optional, helps on Windows
    app = g.compile(checkpointer=SqliteSaver(conn))
    return app

# =========================
# 5) Helpers for checkpoint IDs from history (The LangGraph version: 0.6.x)
# =========================
def extract_checkpoint_id(snapshot) -> str | None:
    """
    In this build, the id is stored under:
      snapshot.config["configurable"]["checkpoint_id"]
    """
    cfg = getattr(snapshot, "config", None)
    if isinstance(cfg, dict):
        conf = cfg.get("configurable") or {}
        if isinstance(conf, dict):
            cid = conf.get("checkpoint_id")
            if isinstance(cid, str) and cid:
                return cid
    return None


# =========================
# 6) Run once (create initial checkpoints)
# =========================
def run_initial_flow(app) -> ChatState:
    """
    Start a conversation, let the graph run draft -> review -> END.
    Usually ends with REVIEW: NEEDS_REVISION (HITL stop).
    """
    cfg = {"configurable": {"thread_id": "conversation-001"}}
    initial: ChatState = {
        "messages": [{"role": "user", "content": "Explain Kubernetes simply."}]
    }
    final_state = app.invoke(initial, config=cfg)

    print("=== FINAL STATE (first run) ===")
    for m in final_state["messages"]:
        print(f"{m['role'].upper()}: {m['content']}")
    return final_state


# =========================
# 7) Show recent checkpoints (pretty print)
# =========================
def show_recent_checkpoints(app, keep_last: int = 8) -> List[str]:
    """
    Fetch history for this conversation and print the LAST N checkpoint IDs.
    Returns the list of IDs in chronological order (oldest -> newest).
    """
    cfg = {"configurable": {"thread_id": "conversation-001"}}
    history = list(app.get_state_history(cfg, limit=200))
    if keep_last:
        history = history[-keep_last:]

    print("\n=== CHECKPOINTS (oldest -> newest) ===")
    ids: List[str] = []
    for i, snap in enumerate(history):
        cid = extract_checkpoint_id(snap)
        ids.append(cid or "<no-id>")
        print(f"{i}. checkpoint_id={ids[-1]}  node=unknown")  # node not stored in this build
    return ids


# =========================
# 8) Time-travel (rewind) and branch (second pass)
# =========================
def rewind_and_branch_from_id(app, checkpoint_id: str):
    """
    Resume from the given checkpoint and add a new user message
    nudging the LLM to improve clarity. Then print the final messages.
    """
    cfg = {
        "configurable": {
            "thread_id": "conversation-001",
            "checkpoint_id": checkpoint_id
        }
    }
    branched = app.invoke(
        {"messages": [{"role": "user", "content": "Please revise to be even clearer for a beginner."}]},
        config=cfg,
    )

    print("\n=== BRANCHED FINAL STATE (after time travel) ===")
    for m in branched["messages"]:
        print(f"{m['role'].upper()}: {m['content']}")


# =========================
# 9) Main 
# =========================
def main():
    # Build the app (graph + checkpointer)
    app = build_graph()

    # First run: typically ends with REVIEW: NEEDS_REVISION (HITL stop)
    run_initial_flow(app)

    # Show the most recent checkpoint IDs (easy to pick from)
    ids = show_recent_checkpoints(app, keep_last=8)

    # Collect indices that actually have usable IDs.
    usable = [c for c in ids if c and c != "<no-id>"]
    if not usable:
        print("\n(No rewind: no usable checkpoint IDs found.)")
        return

    # ---- one-line prompt (with a friendly default) ----
    # Default to the latest usable checkpoint if the user just hits Enter.
    choice = input(f"\nEnter checkpoint index to resume from (0–{len(ids)-1}, default {usable[-1]}): ").strip()
    idx = usable[-1] if choice == "" else (int(choice) if choice.isdigit() else None)

    # Basic validation
    if idx is None or not (0 <= idx < len(ids)) or ids[idx] in (None, "<no-id>"):
        print("Invalid choice. Exiting.")
        return
    
    # Resume FROM the chosen checkpoint (the review stop), prompting a better draft
    rewind_and_branch_from_id(app, ids[idx])


# Run
if __name__ == "__main__":
    assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY in your environment."
    main()
