"""
Streamlit web UI for the Autonomous ML Analyst Agent.

Lets you pick a preset dataset (or upload your own CSV) and watch the agent
reason through all 6 nodes in real-time with live progress indicators.

Run:  streamlit run app.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous ML Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; }
h1 { color: #58a6ff; }
h2 { color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
h3 { color: #56d364; }
.node-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.node-header {
    font-size: 15px;
    font-weight: 700;
    color: #58a6ff;
    margin-bottom: 6px;
}
.metric-box {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}
.warning-box {
    background: #2d1a0a;
    border: 1px solid #d29922;
    border-radius: 6px;
    padding: 10px;
    color: #e3b341;
}
.leakage-box {
    background: #2d1215;
    border: 1px solid #f85149;
    border-radius: 6px;
    padding: 10px;
    color: #ff7b72;
}
.ok-box {
    background: #0f2a1a;
    border: 1px solid #3fb950;
    border-radius: 6px;
    padding: 10px;
    color: #56d364;
}
code { background: #21262d; padding: 2px 6px; border-radius: 4px; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 ML Analyst Agent")
    st.markdown("**LangGraph · Groq · scikit-learn**")
    st.markdown("---")

    PRESETS = {
        "Customer Churn — Classification (Clean)": "dataset_01_classification.csv",
        "Housing Prices — Regression (Clean)": "dataset_02_regression.csv",
        "Employee Attrition — Messy Data (20% missing)": "dataset_03_messy.csv",
        "Customer Churn — Leakage Trap (Tricky)": "dataset_04_leakage.csv",
    }

    mode = st.radio("Dataset source", ["Use preset dataset", "Upload my own CSV"])

    if mode == "Use preset dataset":
        chosen = st.selectbox("Choose dataset", list(PRESETS.keys()))
        csv_path = DATA_DIR / PRESETS[chosen]
        run_name = chosen
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        csv_path = None
        run_name = "Custom Dataset"
        if uploaded:
            tmp = ROOT / "data" / "uploaded_temp.csv"
            tmp.write_bytes(uploaded.read())
            csv_path = tmp
            run_name = uploaded.name.replace(".csv", "").replace("_", " ").title()

    st.markdown("---")
    import os
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        st.success("Groq API: Connected")
        st.caption("Using llama-3.3-70b-versatile")
    else:
        st.warning("Groq API: Not set")
        st.caption("Running in data-driven mock mode")

    run_btn = st.button("Run Agent", type="primary", use_container_width=True, disabled=(csv_path is None))

# ── Main area ──────────────────────────────────────────────────
st.markdown("# 🤖 Autonomous ML Analyst Agent")
st.markdown(
    "Drops a raw CSV → autonomously discovers the target, engineers features, "
    "trains 3 models, and critiques the results. Powered by **LangGraph** + **Groq Llama-3.3-70B**."
)

# Show dataset preview before running
if csv_path and Path(csv_path).exists():
    df_preview = pd.read_csv(csv_path)
    st.markdown("### Dataset Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df_preview.shape[0]:,}")
    col2.metric("Columns", df_preview.shape[1])
    col3.metric("Missing values", df_preview.isna().sum().sum())
    st.dataframe(df_preview.head(8), use_container_width=True)

if not run_btn:
    st.info("Select a dataset in the sidebar and click **Run Agent** to start.")
    st.stop()

# ── Run the agent ──────────────────────────────────────────────
from agent.graph import agent_graph  # noqa: E402

st.markdown("---")
st.markdown(f"## Running: *{run_name}*")

initial_state = {
    "dataset_path": str(csv_path),
    "run_name": run_name,
    "transcript": [],
}

NODE_LABELS = [
    ("DATA PROFILING",         "Inspecting columns, distributions, missing values, correlations"),
    ("PROBLEM FRAMING",        "Deciding target column and problem type (classification vs regression)"),
    ("FEATURE ENGINEERING",    "Planning + executing encoding, scaling, imputation via sklearn"),
    ("MODEL SELECTION",        "Training 3 candidate models with 5-fold cross-validation"),
    ("CRITIQUE",               "Flagging overfitting, data leakage, class imbalance"),
    ("REPORT GENERATION",      "Writing plain-English analysis report"),
]

progress_bar = st.progress(0, text="Starting agent...")
node_placeholders = []
for i, (label, desc) in enumerate(NODE_LABELS):
    with st.expander(f"Node {i+1}/6 — {label}", expanded=False):
        ph = st.empty()
        ph.markdown(f"*{desc}* — waiting...")
        node_placeholders.append(ph)

result_placeholder = st.empty()

# Monkey-patch nodes to stream output into Streamlit
import agent.nodes as _nodes  # noqa: E402

_original_node_fns = {
    "profile_data": _nodes.node_profile_data,
    "frame_problem": _nodes.node_frame_problem,
    "engineer_features": _nodes.node_engineer_features,
    "select_models": _nodes.node_select_and_train_models,
    "critique": _nodes.node_critique_results,
    "generate_report": _nodes.node_generate_report,
}

_node_order = list(_original_node_fns.keys())


def _make_wrapper(original_fn, node_idx):
    def wrapper(state):
        progress_bar.progress(
            (node_idx) / len(NODE_LABELS),
            text=f"Running Node {node_idx+1}/6: {NODE_LABELS[node_idx][0]}..."
        )
        node_placeholders[node_idx].markdown("*Running...*")
        result = original_fn(state)
        # Extract the new text this node added to transcript
        new_entries = result.get("transcript", [])
        if new_entries:
            last_entry = new_entries[-1]
            node_text = last_entry.split("\n", 2)[-1] if "\n" in last_entry else last_entry
            node_placeholders[node_idx].markdown(node_text[:8000])
        progress_bar.progress(
            (node_idx + 1) / len(NODE_LABELS),
            text=f"Completed: {NODE_LABELS[node_idx][0]}"
        )
        return result
    return wrapper


for i, (key, fn) in enumerate(_original_node_fns.items()):
    setattr(_nodes, f"node_{key}" if not key.startswith("node") else key,
            _make_wrapper(fn, i))

# Rebuild graph with wrapped nodes
from langgraph.graph import StateGraph, END, START  # noqa: E402
from agent.state import AgentState  # noqa: E402

g = StateGraph(AgentState)
g.add_node("profile_data",    _make_wrapper(_nodes.node_profile_data, 0))
g.add_node("frame_problem",   _make_wrapper(_nodes.node_frame_problem, 1))
g.add_node("engineer_features", _make_wrapper(_nodes.node_engineer_features, 2))
g.add_node("select_models",   _make_wrapper(_nodes.node_select_and_train_models, 3))
g.add_node("critique",        _make_wrapper(_nodes.node_critique_results, 4))
g.add_node("generate_report", _make_wrapper(_nodes.node_generate_report, 5))
g.add_edge(START, "profile_data")
g.add_edge("profile_data", "frame_problem")
g.add_edge("frame_problem", "engineer_features")
g.add_edge("engineer_features", "select_models")
g.add_edge("select_models", "critique")
g.add_edge("critique", "generate_report")
g.add_edge("generate_report", END)
live_graph = g.compile()

with st.spinner("Agent running..."):
    try:
        final_state = live_graph.invoke(initial_state)
    except Exception as e:
        st.error(f"Agent error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

progress_bar.progress(1.0, text="Analysis complete!")

# ── Results dashboard ──────────────────────────────────────────
st.markdown("---")
st.markdown("## Results")

target_col = final_state.get("target_col", "?")
problem_type = final_state.get("problem_type", "?")
best_model = final_state.get("best_model", "?")
best_score = final_state.get("best_score", 0.0)
leakage = final_state.get("leakage_warning", False)
overfit = final_state.get("overfitting_warning", False)
metric = "F1-score" if problem_type == "classification" else "R²"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target column", target_col)
c2.metric("Problem type", problem_type.upper())
c3.metric("Best model", best_model.replace("Classifier", "").replace("Regressor", ""))
c4.metric(metric, f"{best_score:.4f}")

# Warnings
if leakage:
    st.markdown(
        '<div class="leakage-box">🚨 <strong>DATA LEAKAGE WARNING</strong> — '
        'one or more features are suspiciously correlated with the target. '
        'Scores are artificially inflated. See Critique tab for details.</div>',
        unsafe_allow_html=True,
    )
elif overfit:
    st.markdown(
        '<div class="warning-box">⚠️ <strong>OVERFITTING WARNING</strong> — '
        'significant train/test gap detected. Model complexity should be reduced.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="ok-box">✅ No critical issues detected — results appear reliable.</div>',
        unsafe_allow_html=True,
    )

# Model comparison table
results = final_state.get("model_results", {})
if results:
    st.markdown("### Model Comparison")
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "CV Score": f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
            "Test Score": f"{r['test_score']:.4f}",
            "Train Score": f"{r.get('train_score', 0):.4f}",
            "Train Time": f"{r['train_time_s']:.2f}s",
            "Best": "★" if name == best_model else "",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

# Full report
st.markdown("### Final Report")
report = final_state.get("final_report", "")
st.markdown(report)

# Download buttons
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download Report (MD)", report, file_name=f"{run_name}_report.md", mime="text/markdown"
    )
with col_b:
    transcript = "\n\n---\n\n".join(final_state.get("transcript", []))
    st.download_button(
        "Download Full Transcript (MD)", transcript,
        file_name=f"{run_name}_transcript.md", mime="text/markdown"
    )

st.caption("Autonomous ML Analyst Agent · LangGraph + Groq + scikit-learn · github.com/udayvimal")
