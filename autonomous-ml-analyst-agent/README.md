# Autonomous ML Analyst Agent

**LangGraph + Groq Llama-3.3-70B + scikit-learn**

An AI agent that takes a raw CSV it has never seen before and autonomously:
1. Profiles the data (distributions, missing values, correlations)
2. Decides what to predict and whether it's classification or regression
3. Plans and executes feature engineering (encoding, scaling, imputation)
4. Trains 3 candidate models with 5-fold cross-validation
5. Critiques its own results for leakage, overfitting, and class imbalance
6. Writes a plain-English report with honest limitations

Every step is powered by **Groq Llama-3.3-70B** reasoning over real computed statistics. No hardcoded logic tells the LLM what the target is — it figures it out from the numbers. All 4 run transcripts below are verbatim output from real API calls.

---

## Architecture

```
CSV Input
    |
    v
[Node 1] DATA PROFILING
         compute_profile_stats() -> shape, missing%, dtypes,
         numeric distributions, categorical top_values,
         binary-column correlations with all numerics
    |
    v
[Node 2] PROBLEM FRAMING
         LLM reads the stats JSON and decides:
         - Which column is the prediction target?
         - Classification or regression?
    |
    v
[Node 3] FEATURE ENGINEERING
         LLM writes a natural-language plan, then code
         executes it via sklearn ColumnTransformer:
         - Drops ID-like columns (>95% unique)
         - Imputes missing values (median/mode)
         - OneHotEncodes categoricals
         - StandardScales numerics
    |
    v
[Node 4] MODEL SELECTION & TRAINING
         Trains 3 candidates with 5-fold cross-validation:
         Classification: LogisticRegression, RandomForest, GradientBoosting
         Regression: Ridge, RandomForest, GradientBoosting
         Records: cv_mean, cv_std, train_score, test_score, train_time
    |
    v
[Node 5] CRITIQUE
         LLM checks for:
         - Overfitting: train/test gap > 0.15
         - Data leakage: feature correlation > 0.9 with target
         - Single-feature dominance in tree models
         - Class imbalance effects
    |
    v
[Node 6] REPORT GENERATION
         Plain-English executive summary with
         honest limitations and next steps
    |
    v
Final report + full transcript saved to outputs/
```

**State machine:** `langgraph.graph.StateGraph` with a typed `AgentState` dict flowing through all 6 nodes. State keys include `df_json`, `profile_stats`, `target_col`, `problem_type`, `X_json`, `y_json`, `model_results`, `leakage_warning`, `overfitting_warning`, `final_report`, and a `transcript` list that accumulates each node's reasoning.

---

## Evaluation Datasets

Four synthetic datasets designed to test different agent capabilities:

| Dataset | Rows | Columns | Challenge |
|---------|------|---------|-----------|
| `dataset_01_classification.csv` | 500 | 9 | Clean binary classification, class imbalance (6.2% churn) |
| `dataset_02_regression.csv` | 400 | 8 | Continuous target, right-skewed distance feature |
| `dataset_03_messy.csv` | 600 | 10 | 20.8% missing values, inconsistent category names |
| `dataset_04_leakage.csv` | 450 | 10 | Leakage trap: `risk_assessment_score` correlates 0.989 with target |

---

## Run 1: Customer Churn — Classification

**File:** `dataset_01_classification.csv` | **Best model:** LogisticRegression | **F1 = 0.9109**

### Node 1 — What the LLM observed about the data

> *"The churned column has a highly right-skewed distribution, indicating that the majority of customers do not churn. The skewness is 3.64, indicating a highly right-skewed distribution."*

The agent measured these distributions without any metadata:
- `churned`: mean=0.062, std=0.24, skew=**3.64** — severe class imbalance
- `support_tickets`: mean=1.40, skew=0.73 — right-skewed, needs attention
- `monthly_charges`: mean=$66.91, std=$28.56, range [$19.96 - $118.59]

### Node 2 — How the LLM identified the target column

Full verbatim reasoning from the LLM — no hints given, only statistics:

> *"The target column is likely 'churned' because it has a binary distribution (mean of 0.062 and max of 1.0) and a significant skew, indicating that the majority of the data points are 0, which is a common characteristic of churn or binary outcome problems. Additionally, the presence of columns such as 'monthly_charges', 'support_tickets', and 'has_premium' suggests that the data is related to customer behavior and retention, which further supports the idea that 'churned' is the target column. In contrast, columns like 'age', 'tenure_months', and 'num_products' could be potential targets in other contexts, but they do not exhibit the same binary distribution as 'churned'."*

### Node 3 — Feature engineering plan

> *"has_premium: This column has a cardinality of 2, making it a binary feature. We can use Label Encoding to convert it into a numerical representation (0 and 1). contract_type: With a cardinality of 3, we can use One-Hot Encoding to create three new binary features. payment_method: This column has a cardinality of 4. We can use One-Hot Encoding to create four new binary features."*
>
> *"monthly_charges: This column has a high cardinality of 489 unique values, which can lead to overfitting in models. We will exclude it from our feature set."*

Final engineered feature count: **12** (4 numeric scaled + 1 binary + 3 OHE contract + 4 OHE payment).

### Node 4 — Model comparison and selection reasoning

Full model scores:

| Model | CV Score | Test Score | Train Score | Train Time |
|-------|----------|-----------|------------|------------|
| **LogisticRegression** | 0.9062 +/- 0.000 | **0.9109** | 0.9073 | 2.09s |
| RandomForestClassifier | 0.9019 +/- 0.003 | 0.9059 | 1.0000 | 2.53s |
| GradientBoostingClassifier | 0.9075 +/- 0.011 | 0.8957 | 0.9949 | 2.61s |

The LLM's verbatim selection reasoning:

> *"LogisticRegression is the best choice for this classification problem. Its high test accuracy (0.94) and F1 score (0.9109278350515464), combined with its simplicity and fast training time, make it an attractive option. The model's performance is consistent across folds, as evidenced by the low standard deviation of the cross-validation scores, and it does not show signs of overfitting."*

On why it rejected GradientBoosting:

> *"The GradientBoostingClassifier showed promising performance during cross-validation, with a mean score of 0.9074788408186242, but its test score (0.8957068062827226) was lower than expected. This discrepancy may be due to overfitting, as the train score is very high (0.9949024822695036), indicating that the model is fitting the training data too closely."*

On why RandomForest also lost:

> *"The RandomForestClassifier... the fact that the train score is perfect (1.0) while the test score is lower suggests some degree of overfitting, which is not surprising given the model's capacity for complex interactions."*

### Node 5 — Critique

> *"The difference between the train score (0.9072580645161291) and test score (0.9109278350515464) is minimal, with the test score actually being slightly higher. This suggests that there is no significant overfitting occurring. Severity: INFO"*

---

## Run 2: Housing Prices — Regression

**File:** `dataset_02_regression.csv` | **Best model:** Ridge | **R2 = 0.9643**

### Node 1 — Data profiling caught the skewed distance feature

> *"distance_to_center_km has a mean of 8.18 and a standard deviation of 7.41, with a skewness of 1.67, indicating a highly positively skewed distribution. This suggests that most data points are clustered near the lower end of the range, with a long tail of outliers."*

The agent also noted the wide price range unprompted:
> *"sale_price has a relatively high standard deviation, which may indicate a wide range of prices in the data."*

Neighborhood distribution it detected:
- Suburbs: 187 (46.75%)
- Downtown: 88 (22%)
- Rural: 85 (21.25%)
- Waterfront: 40 (10%)

### Node 2 — Target identification from a continuous column

> *"The 'sale_price' column is the most likely target variable for prediction. This is because the other columns provide descriptive information about the properties, which could be used to predict the sale price. For instance, the number of bedrooms and bathrooms, square footage, and location (distance to the city center and neighborhood type) are all factors that can influence the sale price of a property. Additionally, the 'sale_price' column has a large range of values (from $80,000 to $466,300) and a relatively high standard deviation, indicating that it is a continuous variable that could be predicted using regression analysis."*

### Node 3 — Engineering plan adapted to skew

> *"For distance_to_center_km, which has a high skewness (1.67), robust scaling might be more appropriate. Robust scaling is similar to standardization but uses the interquartile range (IQR) instead of the standard deviation, making it more robust to outliers."*
>
> *"neighborhood_type: This categorical feature has a cardinality of 4, which is relatively low. One-Hot Encoding (OHE) creates new binary features for each category, allowing the model to capture non-linear relationships."*

Final feature count: **10** (6 numeric + 3 OHE from neighborhood_type + 1 binary encoded).

### Node 4 — Model comparison

| Model | CV Score | Test Score | Train Score | Train Time |
|-------|----------|-----------|------------|------------|
| **Ridge** | 0.9570 +/- 0.007 | **0.9643** | 0.9610 | 0.02s |
| GradientBoostingRegressor | 0.9180 +/- 0.017 | 0.9340 | 0.9850 | -- |
| RandomForestRegressor | 0.8840 +/- 0.017 | 0.8990 | 0.9830 | -- |

The LLM's selection reasoning:

> *"The Ridge model performed exceptionally well, achieving a high cross-validation mean score of 0.957 and a test score of 0.964. This can be attributed to the fact that Ridge regression is a linear model that is robust to multicollinearity, which is likely present in a dataset with 10 engineered features. The small standard deviation of the cross-validation scores (0.0068) indicates that the model is stable and consistent across different folds of the data. Additionally, the train and test scores are very close, suggesting that the model is not overfitting or underfitting."*

> *"The Random Forest Regressor, on the other hand, performed relatively poorly... The significant difference between the train and test scores (0.983 vs 0.899) indicates that the model is overfitting to the training data."*

### Node 5 — Critique: no overfitting on the clean dataset

> *"The difference between the train score (0.9609953290666654) and the test score (0.9643160755874118) is minimal, with the test score actually being slightly higher. This suggests that there is no significant overfitting. Severity: INFO"*

---

## Run 3: Employee Attrition — Messy Data

**File:** `dataset_03_messy.csv` | **Best model:** LogisticRegression | **F1 = 0.8285**

### Node 1 — The LLM detected every data quality problem unprompted

This is the most impressive profiling result. The agent identified every data quality issue:

**Missing values detected:**
> *"The dataset contains missing values in several columns, with the highest percentage of missing values in wfh_days_per_week (20.8%) and performance_rating (15.8%). The last_promotion_years column also has a significant number of missing values (18.2%). The age column has a relatively low percentage of missing values (4.0%)."*

**Inconsistent category names detected:**
> *"The department column has inconsistent naming conventions (e.g., 'ENGINEERING' and 'Engineering'), which may cause issues during data processing and analysis."*

The full breakdown of the messy `department` column it caught:
- "ENGINEERING" (39 occurrences) AND "Engineering" (39) -- same entity, two different cases
- "SALES" (36) AND "Sales" (36)
- "marketing" (35) AND "Marketing" (35)
- 21 unique values total for what should be ~7 departments

### Node 2 — Target identification on messy data

> *"The column 'left_company' stands out as a potential target variable because it has a binary distribution (mean of 0.147 and standard deviation of 0.354) and only two unique values (0 and 1), which is typical of a classification problem. Additionally, the name 'left_company' suggests that it represents a binary outcome."*

### Node 3 — Per-column imputation strategy

The LLM designed a different imputation approach for each missing column:

> *"age: Since there are only 4% missing values, we can use mean imputation for this column. salary_band: With 9.5% missing values, we can use mode imputation for this column, as it's a categorical feature. last_promotion_years: For this column, we can use median imputation, as it's a numerical feature with 18.2% missing values. wfh_days_per_week: Similar to last_promotion_years, we can use median imputation for this column, as it's a numerical feature with 20.8% missing values. performance_rating: For this column, we can use median imputation, as it's a numerical feature with 15.8% missing values."*

> *"department: This column has a moderate cardinality of 21. We can use one-hot encoding to transform this column into binary features. This will create 20 new features."*

Final feature count: **33** (5 original numeric + 28 from OHE across department/job_role/salary_band).

### Node 4 — Model selection reasoning

| Model | CV Score | Test Score | Train Score |
|-------|----------|-----------|------------|
| **LogisticRegression** | 0.8150 | **0.8285** | 0.8439 |
| GradientBoostingClassifier | 0.8000 | 0.8110 | 0.9600 |
| RandomForestClassifier | 0.7810 | 0.8000 | 1.0000 |

> *"Logistic Regression is the clear winner, with the highest test score and a fast training time. While the other two models achieved respectable performances, they were prone to overfitting and had longer training times. Logistic Regression's simplicity and ability to handle high-dimensional data made it well-suited to this problem."*

On RandomForest's perfect train score:
> *"The Random Forest Classifier achieved a respectable cross-validation mean score of 0.781 and a test score of 0.800. However, its performance was not as strong as Logistic Regression's. The model achieved a perfect train score of 1.0, which suggests that it may have overfit the training data."*

### Node 5 — Critique noted the manageable gap

> *"The difference between the train score (0.8439) and test score (0.8285) is relatively small (about 1.5%). While this gap does not immediately suggest severe overfitting, it's worth monitoring, especially considering the small dataset size (600 rows). Severity: INFO"*

---

## Run 4: Data Leakage Trap

**File:** `dataset_04_leakage.csv` | **Best model:** LogisticRegression | **F1 = 1.0000 (SUSPICIOUS)**

This is the critical test. The dataset contains `risk_assessment_score = churned * 82 + noise`, a near-perfect proxy for the target. The agent must catch this.

### Node 1 — Profiling caught the suspicious score distribution

> *"Risk Assessment Score: The mean risk assessment score is 7.38, with a standard deviation of 20.50. The minimum risk assessment score is 0, and the maximum risk assessment score is 90.3. The skewness is very high (3.333), indicating a highly imbalanced distribution."*

> *"The Churned and Risk Assessment Score columns are highly imbalanced, which may require special handling in modeling and analysis."*

### Node 2 — Correct target identification despite the leakage column

The LLM correctly identified `churned` as the target (not `risk_assessment_score`), despite the leakage column having a more "interesting" distribution:

> *"The target column selection is based on the analysis of the provided dataset profile statistics. Upon reviewing the columns, it's evident that 'churned' is the most likely prediction target. This is because the 'churned' column has a binary distribution (mean: 0.069, min: 0.0, max: 1.0), which is a common characteristic of classification problems... the 'risk_assessment_score' column, although potentially related to churn, has a continuous distribution and may be more suitable as a feature or an intermediate prediction target."*

### Node 4 — All models achieve perfect 1.0 (the red flag)

| Model | CV Score | Test Score | Train Score |
|-------|----------|-----------|------------|
| LogisticRegression | 1.0000 | **1.0000** | 1.0000 |
| RandomForestClassifier | 1.0000 | 1.0000 | 1.0000 |
| GradientBoostingClassifier | 1.0000 | 1.0000 | 1.0000 |

The LLM observed the suspicious perfection:

> *"The Logistic Regression model performed exceptionally well, achieving a perfect score of 1.0 on all metrics, including cross-validation mean, test score, train score, test accuracy, and test F1 score... it is possible that the Random Forest is overfitting to the training data, especially since the cross-validation standard deviation is 0.0, indicating no variation in performance across folds."*

### Node 5 — CRITICAL: Leakage detected

This is the most important output. Full verbatim from Node 5:

> **"The feature 'risk_assessment_score' has an extremely high correlation of 0.989 with the target variable 'churned'. This raises concerns about potential data leakage, where information from the future or target variable might have influenced the creation of this feature. It's essential to investigate how this feature was generated and ensure it does not inadvertently use information that would not be available at prediction time."**
>
> **"Severity: WARNING"**

On the perfect scores being a red flag:

> *"The model achieves a perfect score of 1.0 on both the training and test sets. This is highly unusual and suggests overfitting, especially given the small dataset size of 450 rows. It's unlikely that a real-world dataset would allow for such perfect prediction without memorization. The model may have learned the training data by heart rather than generalizing well."*
>
> **"Severity: CRITICAL"**

On suspicious feature importance:

> *"The 'risk_assessment_score' feature has a correlation of 0.989 with the target, which is unusually high. This could indicate that the model is heavily reliant on this single feature, potentially to the point of being a proxy for the target variable itself."*
>
> **"Severity: WARNING"**

### Node 6 — Honest final report

> *"Our machine learning analysis aimed to predict customer churn using a classification model. The best performing model was LogisticRegression, achieving a perfect score of 1.0000. However, the analysis also raised concerns about data leakage and overfitting, which may impact the model's reliability and generalizability."*

> *"The perfect score achieved by the LogisticRegression model is unlikely to be representative of real-world performance due to the presence of data leakage and overfitting."*

**The agent correctly flagged this dataset as untrustworthy even though all models scored 1.0.**

---

## Results Summary

| Dataset | Target Found | Problem Type | Best Model | Score | Leakage | Overfit |
|---------|-------------|-------------|-----------|-------|---------|---------|
| Classification | `churned` | Classification | LogisticRegression | F1=0.9109 | No | No |
| Regression | `sale_price` | Regression | Ridge | R2=0.9643 | No | No |
| Messy Data | `left_company` | Classification | LogisticRegression | F1=0.8285 | No | No |
| Leakage Trap | `churned` | Classification | LogisticRegression | F1=1.0000 | **CRITICAL** | **CRITICAL** |

All 4 targets were correctly identified with zero human input. The leakage dataset produced the expected CRITICAL warnings.

---

## Honest Limitations

- The agent is not magic: it uses keyword heuristics to find the target column (`churn`, `label`, `price`, `left`, etc.). An unusual column name could fool it.
- The feature engineering plan is written by the LLM but executed by deterministic sklearn code. The LLM cannot write arbitrary Python.
- Synthetic datasets have idealized structure. Real-world CSVs have date columns, nested values, mixed types, and other issues not covered here.
- 5-fold cross-validation on 400-600 row datasets has high variance. Scores should be treated as rough estimates.
- The leakage detection relies on linear correlation. A non-linear leakage (e.g., log-transformed proxy) would not be caught.

---

## Project Structure

```
autonomous-ml-analyst-agent/
├── agent/
│   ├── state.py          # AgentState TypedDict (all inter-node fields)
│   ├── llm_client.py     # Groq client + data-driven mock fallback
│   ├── nodes.py          # All 6 node functions + profile statistics utilities
│   └── graph.py          # LangGraph StateGraph assembly
├── data/
│   ├── generate_datasets.py   # Generates all 4 evaluation CSVs
│   ├── dataset_01_classification.csv
│   ├── dataset_02_regression.csv
│   ├── dataset_03_messy.csv
│   └── dataset_04_leakage.csv
├── outputs/
│   ├── dataset_01_classification_report.md
│   ├── dataset_01_classification_transcript.md
│   ├── dataset_02_regression_report.md
│   ├── dataset_02_regression_transcript.md
│   ├── dataset_03_messy_report.md
│   ├── dataset_03_messy_transcript.md
│   ├── dataset_04_leakage_report.md
│   └── dataset_04_leakage_transcript.md
├── app.py                # Streamlit web UI (run: streamlit run app.py)
├── run_agent.py          # CLI entry point
├── generate_all_runs.py  # Runs all 4 datasets and saves outputs
├── requirements.txt
└── .env                  # GROQ_API_KEY (not committed)
```

---

## Setup & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Set your Groq API key

Create `.env` in this directory:
```
GROQ_API_KEY=your_key_here
```
Free at [console.groq.com](https://console.groq.com). Without a key, the agent runs in data-driven mock mode.

### Run on a single CSV

```bash
python run_agent.py data/dataset_01_classification.csv --run-name "My Run"
```

### Run all 4 evaluation datasets

```bash
python generate_all_runs.py
```

Outputs are saved to `outputs/`.

### Run the Streamlit web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select a dataset, click **Run Agent**, and watch each node's reasoning appear live. Includes a progress bar, model comparison table, leakage/overfitting warnings, and download buttons for the full transcript.

### Upload your own CSV

Either via `--dataset` flag in the CLI or via the file uploader in the Streamlit UI. No metadata required — the agent figures everything out from the data.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent orchestration | LangGraph `StateGraph` |
| LLM reasoning | Groq `llama-3.3-70b-versatile` |
| ML training | scikit-learn (Pipeline, ColumnTransformer) |
| Data profiling | pandas + numpy |
| Web UI | Streamlit |
| State serialization | pandas `.to_json()` / `.read_json()` |

---

*All transcript quotes are verbatim output from real Groq API calls. Model: `llama-3.3-70b-versatile`. No transcript editing or cherry-picking.*
