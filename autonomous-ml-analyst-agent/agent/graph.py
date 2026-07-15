"""
LangGraph state machine for the Autonomous ML Analyst Agent.

Graph structure (linear — each node's output fully informs the next):

  START
    │
    ▼
  profile_data        ← loads CSV, computes statistics
    │
    ▼
  frame_problem       ← decides target column & problem type
    │
    ▼
  engineer_features   ← plans & executes feature transforms
    │
    ▼
  select_models       ← trains 3 models, compares with CV
    │
    ▼
  critique            ← flags leakage, overfitting, imbalance
    │
    ▼
  generate_report     ← writes final plain-English analysis
    │
    ▼
  END
"""

from langgraph.graph import StateGraph, END, START

from .state import AgentState
from .nodes import (
    node_profile_data,
    node_frame_problem,
    node_engineer_features,
    node_select_and_train_models,
    node_critique_results,
    node_generate_report,
)


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("profile_data", node_profile_data)
    g.add_node("frame_problem", node_frame_problem)
    g.add_node("engineer_features", node_engineer_features)
    g.add_node("select_models", node_select_and_train_models)
    g.add_node("critique", node_critique_results)
    g.add_node("generate_report", node_generate_report)

    g.add_edge(START, "profile_data")
    g.add_edge("profile_data", "frame_problem")
    g.add_edge("frame_problem", "engineer_features")
    g.add_edge("engineer_features", "select_models")
    g.add_edge("select_models", "critique")
    g.add_edge("critique", "generate_report")
    g.add_edge("generate_report", END)

    return g.compile()


# Pre-compiled graph — import and use directly
agent_graph = build_graph()
