# LangGraph LLM Time Travel Demo with Checkpoints

## Project Overview
This project demonstrates how to build a **stateful, checkpoint-based workflow** using [LangGraph](https://github.com/langchain-ai/langgraph) where an LLM:
1. Creates a **draft response**.
2. Reviews the draft and marks it as **Approved** or **Needs Revision**.
3. Supports **"Time Travel"** — resuming from a specific checkpoint to improve and re-review the draft.

The workflow can be restarted from **any saved checkpoint**, allowing iterative improvement without starting over.

---

## Goal
1. **Tiny LangGraph flow:** Draft → Review → HITL → END.  
2. **First run:** LLM drafts, LLM reviews → may return `NEEDS_REVISION`.  
3. **Time Travel:** Resume from any checkpoint, LLM updates the draft, reviews again → eventually `APPROVED`.  
4. **Dynamic user interaction:** The script prompts the user to select which checkpoint to resume from.

---

## Features
- **LLM-powered draft and review process**
- **SQLite checkpointing** with `langgraph-checkpoint-sqlite`
- **Human-in-the-loop (HITL)** for revision control
- **Interactive resume from checkpoint** (Time Travel)
- **Clear visualization of the workflow**

---

## Workflow Diagram

![Workflow Diagram](./langgraph_time_travel_llm_diagram.png)

---

## Project Structure
```
.
├── time_travel_sqlite_demo_LLM.py   # Main script (LLM draft + LLM review + time travel)
├── workflow_diagram.png             # Flowchart image used in README/LinkedIn
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

## How it works (high level)
1. - **Nodes**
        - node_create_draft: asks the LLM to produce a clear, beginner-friendly draft.
        - node_review_draft: asks the LLM reviewer to output **APPROVED** or **NEEDS_REVISION** (strict format).
2. - **Routing**
        - draft → review → END (HITL stop after review).
        - Approval happens on a **resume** run, not in the first pass.
3. - **Checkpoints**
        - We use langgraph-checkpoint-sqlite to persist checkpoints and expose IDs.
        - You can resume from any printed checkpoint to branch the conversation.