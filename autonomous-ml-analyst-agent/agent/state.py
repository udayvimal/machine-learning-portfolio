from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────
    dataset_path: str
    run_name: str

    # ── Node 1: Data Profiling ─────────────────────────────────
    df_json: str
    profile_stats: Dict[str, Any]
    profile_text: str

    # ── Node 2: Problem Framing ────────────────────────────────
    target_col: str
    problem_type: str          # "classification" | "regression"
    problem_frame_text: str

    # ── Node 3: Feature Engineering ───────────────────────────
    feature_plan_text: str
    X_json: str                # engineered feature matrix (JSON)
    y_json: str                # target vector (JSON)
    feature_names: List[str]

    # ── Node 4: Model Selection & Training ────────────────────
    model_results: Dict[str, Any]
    model_results_text: str
    best_model: str
    best_score: float

    # ── Node 5: Critique ──────────────────────────────────────
    critique_text: str
    leakage_warning: bool
    overfitting_warning: bool

    # ── Node 6: Final Report ──────────────────────────────────
    final_report: str

    # ── Running transcript (appended by each node) ────────────
    transcript: List[str]
