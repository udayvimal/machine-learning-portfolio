"""
Generates all 4 evaluation datasets and runs the agent on each.
Saves reports and transcripts to the outputs/ directory.

Usage:  python generate_all_runs.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from data.generate_datasets import generate_all, DATA_DIR, DATASETS
from agent.graph import agent_graph

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

RUNS = [
    ("dataset_01_classification", "Customer Churn — Classification (Clean)"),
    ("dataset_02_regression",     "Housing Prices — Regression (Clean)"),
    ("dataset_03_messy",          "Employee Attrition — Messy Data"),
    ("dataset_04_leakage",        "Customer Churn — Leakage Trap (Tricky)"),
]


def main():
    print("=" * 64)
    print("  Generating evaluation datasets...")
    print("=" * 64)
    generate_all()

    for dataset_name, run_name in RUNS:
        csv_path = DATA_DIR / f"{dataset_name}.csv"
        print(f"\n{'#' * 64}")
        print(f"  RUN: {run_name}")
        print(f"{'#' * 64}")

        initial_state = {
            "dataset_path": str(csv_path),
            "run_name": run_name,
            "transcript": [],
        }

        try:
            final_state = agent_graph.invoke(initial_state)
        except Exception as exc:
            print(f"  ERROR on {dataset_name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

        # Save report
        report = final_state.get("final_report", "")
        stem = dataset_name
        report_path = OUTPUTS_DIR / f"{stem}_report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"\n  → Report saved: {report_path}")

        # Save full transcript
        transcript = final_state.get("transcript", [])
        transcript_text = (
            f"# Agent Transcript: {run_name}\n\n"
            + "\n\n---\n\n".join(transcript)
        )
        transcript_path = OUTPUTS_DIR / f"{stem}_transcript.md"
        transcript_path.write_text(transcript_text, encoding="utf-8")
        print(f"  → Transcript saved: {transcript_path}")

    print("\n" + "=" * 64)
    print("  All runs complete. See outputs/ directory.")
    print("=" * 64)


if __name__ == "__main__":
    main()
