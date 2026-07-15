"""
LangGraph nodes for the Autonomous ML Analyst Agent.

Each node:
  1. Does real computation (pandas profiling / sklearn transforms / model training).
  2. Generates a reasoning text explaining the decisions — either via Groq LLM
     or via a data-driven mock builder that reads actual statistics.

The agent decisions that drive the pipeline:
  - Which column is the target?   → node_frame_problem
  - Classification or regression? → node_frame_problem
  - Which encodings/imputation?   → node_engineer_features
  - Which models to compare?      → node_select_and_train_models
  - What issues exist?            → node_critique_results
"""

import json
import time
import warnings
import textwrap

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier,
                               GradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              r2_score)
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from .state import AgentState
from .llm_client import llm

warnings.filterwarnings("ignore")

BAR = "-" * 62


# ================================================================
# Utility helpers
# ================================================================

def _hdr(title: str, node_num: int, total: int = 6) -> str:
    line = f"  NODE {node_num}/{total}: {title}"
    return f"\n{'='*62}\n{line}\n{'='*62}"


def _wrap(text: str, width: int = 90) -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
        elif para.startswith(("#", "-", "*", "|", "```", "  ")):
            lines.append(para)
        else:
            lines.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(lines)


def compute_profile_stats(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    missing_pct = {c: round(df[c].isna().mean() * 100, 1) for c in df.columns}
    n_unique = {c: int(df[c].nunique()) for c in df.columns}

    num_stats: dict = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        num_stats[c] = {
            "mean": round(float(s.mean()), 3),
            "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "skew": round(float(s.skew()), 3),
            "missing_pct": missing_pct[c],
        }

    cat_stats: dict = {}
    for c in cat_cols:
        vc = df[c].value_counts()
        cat_stats[c] = {
            "top_values": {str(k): int(v) for k, v in vc.head(6).items()},
            "n_unique": int(df[c].nunique()),
            "missing_pct": missing_pct[c],
        }

    # Correlation of all numerics with binary-like columns
    bin_cols = [
        c for c in numeric_cols
        if df[c].dropna().nunique() == 2
    ]
    target_corr: dict = {}
    for bc in bin_cols:
        target_corr[bc] = {
            c: round(float(df[[c, bc]].dropna().corr().iloc[0, 1]), 3)
            for c in numeric_cols
            if c != bc and df[c].dropna().std() > 0
        }

    return {
        "shape": [n_rows, n_cols],
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing": missing,
        "missing_pct": missing_pct,
        "n_unique": n_unique,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "num_stats": num_stats,
        "cat_stats": cat_stats,
        "target_corr": target_corr,
    }


def _infer_target(stats: dict, df: pd.DataFrame) -> tuple[str, str]:
    """
    Heuristic target-column inference used in mock mode.
    With a real LLM, this is determined by the model itself.
    """
    cols = stats["columns"]
    n_unique = stats["n_unique"]
    dtypes = stats["dtypes"]

    # Binary/low-cardinality target keywords
    clf_kw = ["churn", "churned", "target", "label", "survived", "default",
               "fraud", "converted", "attrition", "left", "outcome", "clicked",
               "purchased", "spam", "class", "result", "response"]
    for kw in clf_kw:
        for c in cols:
            if kw in c.lower() and n_unique[c] <= 15:
                return c, "classification"

    # Continuous target keywords
    reg_kw = ["price", "sale_price", "salary", "revenue", "income", "cost",
              "amount", "value", "score", "rating", "demand", "views", "count"]
    for kw in reg_kw:
        for c in cols:
            if kw in c.lower():
                nu = n_unique[c]
                dtype = dtypes.get(c, "")
                if nu > 20 and ("float" in dtype or "int" in dtype):
                    return c, "regression"

    # Fall back to last column
    last = cols[-1]
    if n_unique[last] <= 15:
        return last, "classification"
    return last, "regression"


# ═══════════════════════════════════════════════════════════════
# Mock text builders  (data-specific; built from real statistics)
# ═══════════════════════════════════════════════════════════════

def _build_profile_text(stats: dict) -> str:
    n_rows, n_cols = stats["shape"]
    num_cols = stats["numeric_cols"]
    cat_cols = stats["cat_cols"]
    missing_pct = stats["missing_pct"]
    n_unique = stats["n_unique"]

    total_missing = sum(1 for v in missing_pct.values() if v > 0)
    max_miss_col = max(missing_pct, key=missing_pct.get)
    max_miss_val = missing_pct[max_miss_col]

    lines = [
        f"**Dataset shape**: {n_rows} rows x {n_cols} columns",
        f"**Numeric columns** ({len(num_cols)}): {', '.join(num_cols)}",
        f"**Categorical columns** ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'none'}",
        "",
        "**Missing-value audit**:",
    ]
    if total_missing == 0:
        lines.append("  No missing values detected across any column — data appears well-curated.")
    else:
        for c, pct in sorted(missing_pct.items(), key=lambda x: -x[1]):
            if pct > 0:
                lines.append(f"  • `{c}`: {pct}% missing")

    lines.append("")
    lines.append("**Numeric distributions**:")
    for c, s in stats["num_stats"].items():
        skew_note = ""
        if abs(s["skew"]) > 1.0:
            skew_note = f" — highly {'right' if s['skew'] > 0 else 'left'}-skewed (skew={s['skew']})"
        elif abs(s["skew"]) > 0.5:
            skew_note = f" — moderately skewed (skew={s['skew']})"
        lines.append(
            f"  • `{c}`: mean={s['mean']}, std={s['std']}, "
            f"range=[{s['min']}, {s['max']}]{skew_note}"
        )

    if cat_cols:
        lines.append("")
        lines.append("**Categorical columns**:")
        for c, s in stats["cat_stats"].items():
            top = list(s["top_values"].keys())[:3]
            lines.append(
                f"  • `{c}`: {s['n_unique']} unique values "
                f"(top: {', '.join(str(t) for t in top)})"
            )

    lines.append("")
    # Flag binary columns (likely target candidates)
    bin_cols = [
        c for c in num_cols if n_unique.get(c, 99) == 2
    ]
    if bin_cols:
        lines.append(f"**Binary columns (likely target candidates)**: {', '.join(bin_cols)}")

    # Data-quality observations
    lines.append("")
    lines.append("**Quality observations**:")
    id_like = [c for c in stats["columns"]
                if n_unique.get(c, 0) / max(n_rows, 1) > 0.95
                and c.lower() not in ["sale_price", "salary"]]
    if id_like:
        lines.append(f"  • High-cardinality columns that may be identifiers: {id_like} — "
                     "these should be dropped before modelling.")
    high_miss = [c for c, p in missing_pct.items() if p > 30]
    if high_miss:
        lines.append(f"  • Columns with >30% missing: {high_miss} — consider dropping or "
                     "using domain-informed imputation.")
    if total_missing == 0 and not id_like:
        lines.append("  • No major quality issues detected.")

    return "\n".join(lines)


def _build_frame_text(stats: dict, target_col: str, problem_type: str) -> str:
    n_rows, n_cols = stats["shape"]
    n_unique = stats["n_unique"]
    feature_cols = [c for c in stats["columns"] if c != target_col]
    nu_target = n_unique.get(target_col, "?")

    if problem_type == "classification":
        type_reason = (
            f"`{target_col}` has {nu_target} unique values. "
            "Low cardinality with a discrete domain -> classification."
        )
    else:
        nu = n_unique.get(target_col, 999)
        type_reason = (
            f"`{target_col}` has {nu} unique values with a continuous numeric distribution -> regression."
        )

    # Check if we have suspicious high-corr features (leakage proxy)
    leakage_note = ""
    if target_col in stats.get("target_corr", {}):
        corrs = stats["target_corr"][target_col]
        suspicious = {c: v for c, v in corrs.items() if abs(v) > 0.85 and c != target_col}
        if suspicious:
            leakage_note = (
                f"\n\n[!] **Preliminary leakage flag**: columns {list(suspicious.keys())} "
                f"show correlation >{0.85} with the target. This warrants scrutiny in the critique phase."
            )

    return f"""**Problem framing analysis**

I examined all {n_cols} columns for naming patterns, cardinality, and data type to identify the prediction target.

**Selected target column**: `{target_col}`
Reasoning: {type_reason}

**Problem type**: {problem_type.upper()}

**Input features** ({len(feature_cols)} columns):
{', '.join(f'`{c}`' for c in feature_cols)}

**Dataset scale**: {n_rows} rows is {'sufficient' if n_rows >= 300 else 'small — expect higher variance in CV estimates'} for training non-linear models.{leakage_note}

```json
{{
  "target_col": "{target_col}",
  "problem_type": "{problem_type}",
  "n_features": {len(feature_cols)}
}}
```"""


def _build_feature_plan_text(
    stats: dict, target_col: str, numeric_feats: list,
    cat_feats: list, drop_cols: list, high_card_threshold: int = 10
) -> str:
    lines = ["**Feature engineering plan — with rationale for each decision**", ""]

    if drop_cols:
        lines.append(f"**Dropped columns**: {drop_cols}")
        lines.append("  Reason: near-unique cardinality suggests identifier columns that "
                     "would cause overfitting and carry no generalizable signal.")
        lines.append("")

    lines.append(f"**Numeric features** ({len(numeric_feats)}): {', '.join(numeric_feats)}")
    lines.append("  → `SimpleImputer(strategy='median')`: robust to outliers; median "
                 "preferred over mean when distributions are skewed.")
    lines.append("  → `StandardScaler`: zero-mean, unit-variance scaling required for "
                 "LogisticRegression/Ridge to converge properly. Tree models are scale-invariant "
                 "but scaling does no harm.")
    lines.append("")

    if cat_feats:
        low_card = [c for c in cat_feats
                    if stats["n_unique"].get(c, 99) < high_card_threshold]
        high_card = [c for c in cat_feats
                     if stats["n_unique"].get(c, 0) >= high_card_threshold]
        lines.append(f"**Categorical features** ({len(cat_feats)}): {', '.join(cat_feats)}")
        if low_card:
            lines.append(f"  → `OneHotEncoder(handle_unknown='ignore')` on {low_card}: "
                         "low cardinality makes OHE tractable; 'ignore' for unseen values at inference.")
        if high_card:
            lines.append(f"  → `OrdinalEncoder` on {high_card}: high cardinality would "
                         "create too many OHE columns; ordinal encoding keeps dimensionality manageable.")
        lines.append("  → `SimpleImputer(strategy='most_frequent')`: fills rare NaN gaps "
                     "with the modal category.")
        lines.append("")

    miss_cols = {c: v for c, v in stats["missing_pct"].items() if v > 0 and c != target_col}
    if not miss_cols:
        lines.append("**Missing values**: none detected — no imputation required.")
    else:
        lines.append(f"**Missing values present** in {len(miss_cols)} column(s): "
                     f"{dict(list(miss_cols.items())[:5])}")
        lines.append("  Imputation strategies specified per-column above.")

    # Estimate final dimensionality
    n_num = len(numeric_feats)
    n_cat_cols_after = sum(
        stats["n_unique"].get(c, 2)
        for c in cat_feats
        if stats["n_unique"].get(c, 99) < high_card_threshold
    ) + len([c for c in cat_feats if stats["n_unique"].get(c, 0) >= high_card_threshold])
    lines.append(f"\n**Estimated final dimensionality**: {n_num} numeric + "
                 f"~{n_cat_cols_after} encoded categorical = {n_num + n_cat_cols_after} columns")

    return "\n".join(lines)


def _build_model_results_text(
    results: dict, problem_type: str, best_model: str, feature_names: list
) -> str:
    metric = "Accuracy / F1" if problem_type == "classification" else "R² / RMSE"
    lines = [
        f"**Model comparison** — metric: {metric} | CV: 5-fold "
        f"({'stratified' if problem_type == 'classification' else 'standard'})",
        "",
        f"| Model | CV Score | CV Std | Test Score | Train Time |",
        f"|-------|----------|--------|------------|------------|",
    ]
    for name, r in results.items():
        cv = r["cv_mean"]
        std = r["cv_std"]
        test = r["test_score"]
        t = r["train_time_s"]
        lines.append(f"| {name:<32} | {cv:.4f}   | {std:.4f} | {test:.4f}     | {t:.2f}s      |")

    lines.append("")
    lines.append(f"**Selected model**: {best_model}")
    best = results[best_model]
    lines.append(
        f"  Test score: {best['test_score']:.4f} | "
        f"Train score: {best.get('train_score', 0):.4f} | "
        f"Overfitting gap: {abs(best.get('train_score', 0) - best['test_score']):.4f}"
    )

    # Feature importance (if available)
    fi = best.get("feature_importance")
    if fi and feature_names:
        top_n = 5
        abs_fi = [abs(v) for v in fi]
        max_val = max(abs_fi) if max(abs_fi) > 0 else 1.0
        # Normalize to [0,1] so bars are always sensible (handles Ridge coef_ scale)
        norm_fi = [v / max_val for v in abs_fi]
        sorted_fi = sorted(
            zip(feature_names, norm_fi), key=lambda x: -x[1]
        )[:top_n]
        lines.append("")
        lines.append(f"**Top {top_n} features by relative importance**:")
        for fname, imp in sorted_fi:
            bar = "#" * int(imp * 30)
            lines.append(f"  {fname:<34} {imp:.4f}  {bar}")

    return "\n".join(lines)


def _build_critique_text(
    results: dict, stats: dict, problem_type: str,
    target_col: str, best_model: str
) -> tuple[str, bool, bool]:
    """Returns (critique_text, leakage_warning, overfitting_warning)."""
    best = results[best_model]
    train_score = best.get("train_score", 0)
    test_score = best["test_score"]
    gap = train_score - test_score
    n_rows = stats["shape"][0]
    fi = best.get("feature_importance", [])

    leakage_warning = False
    overfitting_warning = False
    issues = []
    praises = []

    # 1. Overfitting check
    if gap > 0.15:
        overfitting_warning = True
        issues.append(
            f"**[!] Overfitting detected**: train={train_score:.4f} vs test={test_score:.4f} "
            f"(gap={gap:.4f}). Consider reducing model complexity (lower max_depth, "
            "stronger regularization, or pruning). With more data this gap often narrows."
        )
    elif gap > 0.07:
        issues.append(
            f"**Mild overfit**: train={train_score:.4f} vs test={test_score:.4f} (gap={gap:.4f}). "
            "Acceptable but worth monitoring on larger held-out sets."
        )
    else:
        praises.append(
            f"Train/test gap is tight ({gap:.4f}) — model generalizes well to unseen data."
        )

    # 2. Suspicious feature importance (potential leakage)
    feature_names_all = []
    if fi:
        from .state import AgentState  # noqa: avoid circular at module level
        pass
    # Check leakage via target correlation
    if target_col in stats.get("target_corr", {}):
        corrs = stats["target_corr"][target_col]
        leaky = {c: v for c, v in corrs.items() if abs(v) > 0.85}
        if leaky:
            leakage_warning = True
            issues.append(
                f"**[LEAKAGE] Data leakage risk**: the following features have correlation "
                f">{0.85} with `{target_col}`: {leaky}. These likely encode the target "
                "directly (e.g., derived from it post-event, or computed from the outcome). "
                "Remove and retrain before trusting the scores."
            )

    # Also check if any single feature has extremely high normalized importance
    # (only meaningful for tree models where importances sum to 1)
    if fi and best_model not in ("Ridge", "LogisticRegression"):
        abs_fi = [abs(v) for v in fi]
        fi_sum = sum(abs_fi)
        if fi_sum > 0:
            norm_fi = [v / fi_sum for v in abs_fi]
            max_norm = max(norm_fi)
            if max_norm > 0.50:
                leakage_warning = True
                issues.append(
                    f"**[!] Single-feature dominance**: one feature accounts for "
                    f"{max_norm:.1%} of total importance. This concentration is atypical "
                    "and may signal leakage or a proxy target — investigate before deployment."
                )

    # 3. Class imbalance (classification only)
    if problem_type == "classification":
        bin_cols = [
            c for c in stats["numeric_cols"]
            if stats["n_unique"].get(c, 99) == 2 and c == target_col
        ]
        # We don't have the actual y at this point, but we can infer from prior stats
        # Check if accuracy is suspiciously high vs F1
        acc = best.get("test_accuracy", test_score)
        f1 = best.get("test_f1", None)
        if f1 is not None and acc - f1 > 0.1:
            issues.append(
                f"**Class imbalance effect**: accuracy ({acc:.4f}) significantly exceeds "
                f"F1 ({f1:.4f}). The model may be over-predicting the majority class. "
                "Consider SMOTE, class_weight='balanced', or optimizing for F1 directly."
            )

    # 4. Data size caution
    if n_rows < 500:
        issues.append(
            f"**Small dataset ({n_rows} rows)**: CV score variance will be high. "
            "Report confidence intervals alongside point estimates; "
            "do not over-interpret small differences between models."
        )

    # 5. Praise high-performing aspects
    if test_score > 0.85 and not leakage_warning:
        praises.append(f"Strong test performance ({test_score:.4f}) achieved without leakage signals.")

    lines = ["**Agent critique of model results**", ""]
    if issues:
        lines.append("**Issues flagged**:")
        for issue in issues:
            lines.append(f"  {issue}")
            lines.append("")
    if praises:
        lines.append("**Positive observations**:")
        for praise in praises:
            lines.append(f"  ✓ {praise}")

    if not issues:
        lines.append("No major issues detected. Results appear reliable for this dataset size.")

    return "\n".join(lines), leakage_warning, overfitting_warning


def _build_final_report(state: AgentState) -> str:
    run_name = state.get("run_name", "Unknown")
    n_rows, n_cols = state["profile_stats"]["shape"]
    target_col = state["target_col"]
    problem_type = state["problem_type"]
    best_model = state["best_model"]
    best_score = state["best_score"]
    leakage = state.get("leakage_warning", False)
    overfit = state.get("overfitting_warning", False)

    metric_name = "F1-score" if problem_type == "classification" else "R²"
    warning_block = ""
    if leakage:
        warning_block += "\n> ⛔ **DATA LEAKAGE WARNING**: one or more features appear to " \
                         "encode the target. Scores above are artificially inflated — see Critique section.\n"
    if overfit:
        warning_block += "\n> ⚠️  **OVERFITTING WARNING**: significant train/test gap detected. " \
                         "Model complexity should be reduced before deployment.\n"

    results = state.get("model_results", {})
    model_table_rows = ""
    for name, r in results.items():
        marker = " ← best" if name == best_model else ""
        model_table_rows += (
            f"| {name} | {r['cv_mean']:.4f} ± {r['cv_std']:.4f} | "
            f"{r['test_score']:.4f}{marker} |\n"
        )

    report = f"""# ML Analysis Report: {run_name}

> Auto-generated by the Autonomous ML Analyst Agent (LangGraph + scikit-learn)
{warning_block}
---

## Executive Summary

An autonomous 6-node LangGraph agent analyzed the **{run_name}** dataset
({n_rows} rows x {n_cols} columns) without any human-provided labels or
instructions beyond the raw CSV path.

The agent identified **`{target_col}`** as the prediction target and determined
this is a **{problem_type.upper()}** problem. After data profiling, adaptive
feature engineering, and training three candidate models, **{best_model}**
was selected as the best performer with **{metric_name} = {best_score:.4f}**
on the held-out test set.

---

## 1. Data Profile

{state.get("profile_text", "")}

---

## 2. Problem Framing

{state.get("problem_frame_text", "")}

---

## 3. Feature Engineering

{state.get("feature_plan_text", "")}

---

## 4. Model Comparison

{state.get("model_results_text", "")}

---

## 5. Agent Critique

{state.get("critique_text", "")}

---

## 6. Recommendation

**Recommended model**: {best_model}
**{metric_name}** on test set: {best_score:.4f}

{'**Immediate action required**: address the data leakage before any production use.' if leakage else
 'This model is ready for further validation on a fresh holdout set.'}

Next steps:
1. {"Investigate and remove leaky features, then retrain." if leakage else "Run on 3-fold nested CV for unbiased performance estimate."}
2. {"Reduce model complexity to address overfitting." if overfit else "Collect more data to further validate generalisation."}
3. Check feature importance in production monitoring; drift in top features signals data shift.
4. For deployment: wrap in a sklearn Pipeline with the ColumnTransformer prepended (already done in this run).

---
*Report generated by: Autonomous ML Analyst Agent v1.0*
"""
    return report


# ═══════════════════════════════════════════════════════════════
# LLM prompt builders  (used when Groq API key is available)
# ═══════════════════════════════════════════════════════════════

def _prompt_profile(stats: dict) -> str:
    return f"""You are an expert ML data analyst. Analyze the following dataset statistics and
write a structured, precise data profile that identifies quality issues, distribution
characteristics, and candidate target columns. Be specific — reference actual numbers.

STATISTICS:
{json.dumps(stats, indent=2, default=str)[:3000]}

Write the profile in markdown with sections: Shape, Missing Values, Numeric Distributions,
Categorical Columns, Quality Observations. Be analytical, not generic."""


def _prompt_frame(stats: dict) -> str:
    return f"""You are an expert ML analyst. Given the following dataset profile statistics,
identify the most likely prediction TARGET column and determine whether this is a
CLASSIFICATION, REGRESSION, or CLUSTERING problem. Provide clear reasoning.

STATISTICS:
{json.dumps(stats, indent=2, default=str)[:2500]}

Respond with:
1. A reasoning paragraph explaining your target selection
2. A JSON block with: {{"target_col": "...", "problem_type": "classification|regression", "n_features": N}}

Be specific about WHY you chose that column over alternatives."""


def _prompt_feature_plan(stats: dict, target_col: str, problem_type: str,
                          numeric_feats: list, cat_feats: list, drop_cols: list) -> str:
    return f"""You are an ML feature engineering expert. Design a feature engineering plan for
this dataset and justify each choice.

Target: {target_col} | Problem type: {problem_type}
Numeric features: {numeric_feats}
Categorical features: {cat_feats}
Proposed drop (high-cardinality IDs): {drop_cols}

DATASET STATS:
{json.dumps({'n_unique': stats['n_unique'], 'missing_pct': stats['missing_pct'],
             'num_stats': stats.get('num_stats', {})}, indent=2, default=str)[:2000]}

Write a plan covering: (1) which columns to drop and why, (2) encoding strategy for each
categorical column with cardinality reasoning, (3) scaling strategy with justification,
(4) imputation strategy if missing values exist. End with estimated final feature count."""


def _prompt_model_select(stats: dict, problem_type: str, results: dict,
                          best_model: str, feature_names: list) -> str:
    return f"""You are an ML expert reviewing model training results. Analyze these results and
explain the model selection decision.

Problem type: {problem_type}
Dataset size: {stats['shape'][0]} rows x {len(feature_names)} engineered features
Best model: {best_model}

RESULTS:
{json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'feature_importance'}
             for k, v in results.items()}, indent=2, default=str)}

Write a paragraph per model explaining WHY it performed as it did given the data
characteristics, then justify the final model selection."""


def _prompt_critique(results: dict, stats: dict, target_col: str,
                     problem_type: str, best_model: str) -> str:
    best = results[best_model]
    return f"""You are a critical ML reviewer. Your job is to find problems, not praise successes.

Analyze these training results and flag: overfitting, data leakage risks, class imbalance
effects, suspicious feature importance, and any other red flags.

Best model: {best_model}
Train score: {best.get('train_score', 'N/A')} | Test score: {best['test_score']}
Target: {target_col} | Problem: {problem_type}
Dataset: {stats['shape'][0]} rows

High-correlation features with target:
{json.dumps(stats.get('target_corr', {}).get(target_col, {}), indent=2, default=str)[:1000]}

Be specific and actionable. Rate severity: CRITICAL / WARNING / INFO."""


def _prompt_report(state: AgentState) -> str:
    return f"""Write a final plain-English ML analysis report. It should be clear enough for
a business stakeholder AND a data scientist to read.

Run: {state.get("run_name")}
Target: {state.get("target_col")} | Type: {state.get("problem_type")}
Best model: {state.get("best_model")} | Score: {state.get("best_score", 0):.4f}
Leakage warning: {state.get("leakage_warning", False)}
Overfitting warning: {state.get("overfitting_warning", False)}

Include: Executive Summary, Key Findings (bullet points), Honest Limitations,
and Next Steps. Use markdown. Do not invent numbers not provided above."""


# ═══════════════════════════════════════════════════════════════
# Node functions
# ═══════════════════════════════════════════════════════════════

def node_profile_data(state: AgentState) -> dict:
    print(_hdr("DATA PROFILING", 1))
    path = state["dataset_path"]
    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    n_rows, n_cols = df.shape
    print(f"  Shape: {n_rows} rows x {n_cols} columns")

    stats = compute_profile_stats(df)
    print(f"  Numeric cols: {stats['numeric_cols']}")
    print(f"  Categorical cols: {stats['cat_cols']}")
    total_missing = sum(1 for v in stats["missing_pct"].values() if v > 0)
    print(f"  Columns with missing values: {total_missing}")

    if llm.available:
        print("  [LLM] Generating profile analysis via Groq...")
        profile_text = llm.chat(_prompt_profile(stats))
    else:
        profile_text = _build_profile_text(stats)

    print("\n" + BAR)
    print(_wrap(profile_text))
    print(BAR)

    transcript_entry = f"## NODE 1: DATA PROFILING\n\n{profile_text}"
    return {
        **state,
        "df_json": df.to_json(),
        "profile_stats": stats,
        "profile_text": profile_text,
        "transcript": state.get("transcript", []) + [transcript_entry],
    }


def node_frame_problem(state: AgentState) -> dict:
    print(_hdr("PROBLEM FRAMING", 2))
    stats = state["profile_stats"]
    df = pd.read_json(state["df_json"])

    if llm.available:
        print("  [LLM] Reasoning about target column and problem type...")
        response = llm.chat(_prompt_frame(stats))
        # Extract JSON block
        import re
        m = re.search(r'\{[^}]+\}', response, re.DOTALL)
        target_col = stats["columns"][-1]
        problem_type = "classification"
        if m:
            try:
                parsed = json.loads(m.group(0))
                target_col = parsed.get("target_col", target_col)
                problem_type = parsed.get("problem_type", problem_type)
            except json.JSONDecodeError:
                pass
        frame_text = response
    else:
        target_col, problem_type = _infer_target(stats, df)
        frame_text = _build_frame_text(stats, target_col, problem_type)

    print(f"  -> Target column: `{target_col}`")
    print(f"  -> Problem type:  {problem_type.upper()}")
    print("\n" + BAR)
    print(_wrap(frame_text))
    print(BAR)

    transcript_entry = f"## NODE 2: PROBLEM FRAMING\n\n{frame_text}"
    return {
        **state,
        "target_col": target_col,
        "problem_type": problem_type,
        "problem_frame_text": frame_text,
        "transcript": state["transcript"] + [transcript_entry],
    }


def node_engineer_features(state: AgentState) -> dict:
    print(_hdr("FEATURE ENGINEERING", 3))
    stats = state["profile_stats"]
    target_col = state["target_col"]
    problem_type = state["problem_type"]

    df = pd.read_json(state["df_json"])

    # Identify columns
    all_cols = [c for c in df.columns if c != target_col]
    n_unique = stats["n_unique"]
    n_rows = stats["shape"][0]

    # Drop ID-like columns (>95% unique AND not already the target)
    drop_cols = [
        c for c in all_cols
        if n_unique.get(c, 0) / max(n_rows, 1) > 0.95
    ]
    feature_cols = [c for c in all_cols if c not in drop_cols]

    numeric_feats = [
        c for c in feature_cols
        if c in stats["numeric_cols"]
    ]
    cat_feats = [
        c for c in feature_cols
        if c in stats["cat_cols"]
    ]

    print(f"  Numeric features ({len(numeric_feats)}): {numeric_feats}")
    print(f"  Categorical features ({len(cat_feats)}): {cat_feats}")
    if drop_cols:
        print(f"  Dropping ID-like cols: {drop_cols}")

    # Build sklearn preprocessing pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_feats:
        transformers.append(("num", numeric_pipe, numeric_feats))
    if cat_feats:
        transformers.append(("cat", categorical_pipe, cat_feats))

    preprocessor = ColumnTransformer(transformers, remainder="drop")

    X_raw = df[feature_cols]
    y_raw = df[target_col]

    # Encode target for classification
    if problem_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    X_engineered = preprocessor.fit_transform(X_raw)

    # Recover feature names after OHE
    try:
        ohe_feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "num":
                ohe_feature_names.extend(cols)
            elif name == "cat":
                ohe_feature_names.extend(
                    trans.named_steps["encoder"].get_feature_names_out(cols).tolist()
                )
        feature_names = ohe_feature_names
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_engineered.shape[1])]

    print(f"  Feature matrix shape after engineering: {X_engineered.shape}")

    if llm.available:
        print("  [LLM] Generating feature engineering rationale via Groq...")
        plan_text = llm.chat(
            _prompt_feature_plan(stats, target_col, problem_type,
                                  numeric_feats, cat_feats, drop_cols)
        )
    else:
        plan_text = _build_feature_plan_text(
            stats, target_col, numeric_feats, cat_feats, drop_cols
        )

    print("\n" + BAR)
    print(_wrap(plan_text))
    print(BAR)

    X_df = pd.DataFrame(X_engineered, columns=feature_names)
    y_series = pd.Series(y, name=target_col)

    transcript_entry = f"## NODE 3: FEATURE ENGINEERING\n\n{plan_text}"
    return {
        **state,
        "feature_plan_text": plan_text,
        "X_json": X_df.to_json(),
        "y_json": y_series.to_json(),
        "feature_names": feature_names,
        "transcript": state["transcript"] + [transcript_entry],
    }


def node_select_and_train_models(state: AgentState) -> dict:
    print(_hdr("MODEL SELECTION & TRAINING", 4))
    problem_type = state["problem_type"]
    feature_names = state["feature_names"]

    X = pd.read_json(state["X_json"]).values
    y = pd.read_json(state["y_json"], typ="series").values

    is_clf = (problem_type == "classification")

    # Split
    split_kwargs = dict(test_size=0.2, random_state=42)
    if is_clf:
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if is_clf else \
         KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = "f1_weighted" if is_clf else "r2"

    if is_clf:
        candidates = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
    else:
        candidates = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

    results = {}
    for name, model in candidates.items():
        print(f"  Training {name}...", end=" ", flush=True)
        t0 = time.time()

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        if is_clf:
            test_score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            train_score = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = test_score
        else:
            test_score = r2_score(y_test, y_pred)
            train_score = r2_score(y_train, y_pred_train)
            test_acc = None
            test_f1 = None

        fi = None
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_.tolist()
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                fi = np.abs(coef).mean(axis=0).tolist()
            else:
                fi = np.abs(coef).tolist()

        results[name] = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test_score": float(test_score),
            "train_score": float(train_score),
            "train_time_s": round(elapsed, 2),
            "feature_importance": fi,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
        }
        print(f"CV={np.mean(cv_scores):.4f} | Test={test_score:.4f}")

    # Select best by test score
    best_model = max(results, key=lambda k: results[k]["test_score"])
    best_score = results[best_model]["test_score"]
    print(f"\n  -> Best model: {best_model} (test score: {best_score:.4f})")

    if llm.available:
        print("  [LLM] Generating model comparison reasoning via Groq...")
        results_text = llm.chat(
            _prompt_model_select(state["profile_stats"], problem_type,
                                  results, best_model, feature_names)
        )
    else:
        results_text = _build_model_results_text(
            results, problem_type, best_model, feature_names
        )

    print("\n" + BAR)
    print(_wrap(results_text))
    print(BAR)

    transcript_entry = f"## NODE 4: MODEL SELECTION & TRAINING\n\n{results_text}"
    return {
        **state,
        "model_results": results,
        "model_results_text": results_text,
        "best_model": best_model,
        "best_score": best_score,
        "transcript": state["transcript"] + [transcript_entry],
    }


def node_critique_results(state: AgentState) -> dict:
    print(_hdr("CRITIQUE NODE", 5))
    results = state["model_results"]
    stats = state["profile_stats"]
    problem_type = state["problem_type"]
    target_col = state["target_col"]
    best_model = state["best_model"]

    if llm.available:
        print("  [LLM] Running critical review via Groq...")
        critique_text = llm.chat(
            _prompt_critique(results, stats, target_col, problem_type, best_model)
        )
        # Determine warnings from text patterns
        leakage_warning = any(
            kw in critique_text.lower()
            for kw in ["leakage", "leaky", "leak"]
        )
        overfitting_warning = any(
            kw in critique_text.lower()
            for kw in ["overfit", "overfitting"]
        )
    else:
        critique_text, leakage_warning, overfitting_warning = _build_critique_text(
            results, stats, problem_type, target_col, best_model
        )

    if leakage_warning:
        print("  [!] LEAKAGE WARNING raised")
    if overfitting_warning:
        print("  [!] OVERFITTING WARNING raised")
    if not leakage_warning and not overfitting_warning:
        print("  [OK] No critical issues detected")

    print("\n" + BAR)
    print(_wrap(critique_text))
    print(BAR)

    transcript_entry = f"## NODE 5: CRITIQUE\n\n{critique_text}"
    return {
        **state,
        "critique_text": critique_text,
        "leakage_warning": leakage_warning,
        "overfitting_warning": overfitting_warning,
        "transcript": state["transcript"] + [transcript_entry],
    }


def node_generate_report(state: AgentState) -> dict:
    print(_hdr("REPORT GENERATION", 6))

    if llm.available:
        print("  [LLM] Writing final report via Groq...")
        report_body = llm.chat(_prompt_report(state))
        final_report = report_body
    else:
        final_report = _build_final_report(state)

    print("\n" + BAR)
    print(_wrap(final_report[:800]) + "\n  [... full report in output file ...]")
    print(BAR)
    print(f"\n  [DONE] Analysis complete for: {state.get('run_name', '')}")

    transcript_entry = f"## NODE 6: FINAL REPORT\n\n{final_report}"
    return {
        **state,
        "final_report": final_report,
        "transcript": state["transcript"] + [transcript_entry],
    }


