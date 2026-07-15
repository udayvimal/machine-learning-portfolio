# Uday Vimal — Machine Learning Portfolio

Python · scikit-learn · Deep Learning · Computer Vision · NLP · Deployment

---

## Key Results

- **Sourcing Fraud Prediction** — gradient boosting classifier detecting procurement fraud patterns; feature-engineered from vendor and transaction data, evaluated with precision/recall at business-relevant thresholds.
- **Customer Churn Prediction** — decision tree model on SME customer churn dataset with JWT-authenticated FastAPI backend; end-to-end from training script to REST prediction endpoint.
- **Demand Forecasting** — stacking regressor (Ridge + Lasso + GradientBoosting) with Optuna hyperparameter tuning on YouTube ad-view data; log-transform target, 10-fold CV.
- **Road Sign Detection** — YOLOv8 fine-tuned on a real-world annotated dataset (Roboflow); evaluated on held-out test split.

---

## About this repository

These projects were built individually during my B.Tech — across coursework, internships, and personal exploration — and collected into this repo as I set up GitHub properly. Some folder names reflect the original working names I used at the time. The three flagship projects listed below are the primary work to look at first; the remaining folders are additional coursework and tutorial-level implementations.

---

## Projects

> **Primary projects to look at first** (marked with ★):

| Folder | Description |
|---|---|
| ★ [Sourcing Fraud Prediction Model](Sourcing-Fraud-Prediction-Model/) | Gradient boosting classifier for procurement fraud detection — feature engineering, threshold tuning, evaluation report |
| ★ [Customer Churn Prediction](User-Authentication-ML-Prediction-System-main/) | Decision tree churn model with FastAPI + JWT auth backend; `train_model.py` trains on `sme_customer_churn.csv` |
| ★ [Demand Forecasting / YouTube Ads View Prediction](Youtube-ads-view-prediction--main/) | Stacking regressor with Optuna tuning; Ridge + Lasso + GBR; 10-fold CV; log-transform pipeline |
| [Image Recognition Chatbot](Image_Recognition_Chatbot-master/) | Vision chatbot: BLIP image captioning + DeiT classification; responds to fabric/style/season queries |
| [Movie Recommendation System](MOVIE-RECOMMENDATION-SYSTEM-main/) | Content-based filtering with cosine similarity + Streamlit dark-mode UI; run `preprocess.py` first |
| [Real-Time Object Detection](REAL-TIME-OBJECT-DETECTION-SYSTEM-main/) | YOLOv8 real-time detection pipeline with OpenCV webcam feed |
| [Road Sign Detection](RoadSignDetection-master/) | YOLOv8 fine-tuned on Roboflow road sign dataset |
| [AI Chatbot](aichatbot-master/) | FAISS + LangChain + Mistral RAG chatbot on PDF documents |
| [Data Summarization](data-summarization-master/) | Flask app using BART (facebook/bart-large-cnn) for abstractive text summarization |
| [Fake News Detection](fakenewspredicton-master/) | TF-IDF + PassiveAggressiveClassifier Flask web app; run `train_model.py` first |
| [Hotel Booking Prediction](HOTEBOOKINGPREDICTION-main/) | Logistic regression model predicting cancellations; full EDA and preprocessing notebook |

---

## Tech Stack

| Area | Technologies |
|---|---|
| ML / Classical | scikit-learn, pandas, NumPy, Optuna, XGBoost, LightGBM |
| Deep Learning | PyTorch, Transformers (BLIP, BART, DeiT) |
| Computer Vision | YOLOv8 (Ultralytics), OpenCV |
| NLP | TF-IDF, FAISS, LangChain, Mistral |
| Deployment | FastAPI, Flask, Streamlit, SQLAlchemy, JWT |
| Visualization | matplotlib, seaborn |

---

## Contact

**Email:** udayvimal08@gmail.com
**GitHub:** [github.com/udayvimal](https://github.com/udayvimal)
**LinkedIn:** [linkedin.com/in/uday-vimal-9a1a3a253](https://linkedin.com/in/uday-vimal-9a1a3a253)
