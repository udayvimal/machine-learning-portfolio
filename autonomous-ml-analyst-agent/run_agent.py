"""
Autonomous ML Analyst Agent — CLI entry point.

Usage:
  python run_agent.py <csv_path> [--run-name "My Run"] [--output results.md]

Examples:
  python run_agent.py data/dataset_01_classification.csv
  python run_agent.py data/dataset_02_regression.csv --run-name "Housing Price Run"
  python run_agent.py your_own_data.csv --output my_analysis.md

The agent will:
  1. Profile the dataset (no metadata required — just a raw CSV)
  2. Decide the target variable and problem type
  3. Engineer features adaptively
  4. Train and compare 3 ML models
  5. Critique the results (flags leakage, overfitting, imbalance)
  6. Write a plain-English analysis report

Set GROQ_API_KEY env var to use real LLM reasoning (Groq / Llama-3.3-70B).
Without it, the agent runs in data-driven mock mode (fully functional ML,
rule-based reasoning text).
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous ML Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable name for this analysis run (default: CSV filename)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the markdown report (default: <csv_stem>_report.md)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_name or csv_path.stem.replace("_", " ").title()
    output_path = Path(args.output) if args.output else csv_path.with_suffix("_report.md")
    if args.output is None:
        output_path = csv_path.parent.parent / "outputs" / f"{csv_path.stem}_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  AUTONOMOUS ML ANALYST AGENT  (LangGraph v1.0)")
    print("=" * 64)
    print(f"  Dataset : {csv_path}")
    print(f"  Run name: {run_name}")
    print(f"  Output  : {output_path}")

    # Late import — lets the LLMClient print its mode indicator after header
    from agent.graph import agent_graph

    initial_state = {
        "dataset_path": str(csv_path),
        "run_name": run_name,
        "transcript": [],
    }

    try:
        final_state = agent_graph.invoke(initial_state)
    except Exception as exc:
        print(f"\nERROR during agent execution: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save report
    report = final_state.get("final_report", "")
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  📄 Report saved → {output_path}")

    # Save full transcript (all 6 node outputs)
    transcript = final_state.get("transcript", [])
    transcript_path = output_path.with_name(output_path.stem + "_transcript.md")
    full_transcript = f"# Full Agent Transcript: {run_name}\n\n" + "\n\n---\n\n".join(transcript)
    transcript_path.write_text(full_transcript, encoding="utf-8")
    print(f"  📋 Transcript saved → {transcript_path}")

    return final_state


if __name__ == "__main__":
    main()
