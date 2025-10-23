# ============================================================
# src/ml_pipeline.py
# Customer Churn Prediction - Machine Learning Pipeline
# Fully Offline (Prefect Stubbed + Noise Suppressed)
# ============================================================

import os
import sys
import json
import types
import warnings
import logging
import queue
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------- Suppress Prefect & Thread Warnings -------------------
warnings.filterwarnings("ignore", category=UserWarning, module="prefect")

def safe_thread_run(original_run):
    """Patch Prefect background threads to suppress harmless _queue.Empty errors."""
    def wrapped_run(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        except queue.Empty:
            logging.debug("Prefect background queue empty (ignored).")
            return None
        except Exception as e:
            logging.error(f"Prefect background thread exception: {e}")
            return None
    return wrapped_run

threading.Thread.run = safe_thread_run(threading.Thread.run)

# ------------------- Prefect Offline Setup -------------------
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_API_ENABLE"] = "false"
os.environ["PREFECT_TEST_MODE"] = "true"
os.environ["PREFECT_LOCAL_MODE"] = "true"
os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
os.environ["PREFECT_ANALYTICS_ENABLED"] = "false"

# Prefect stubs (no network / events)
worker_stub = types.ModuleType("prefect.events.worker")
worker_stub.EventsWorker = type("EventsWorker", (), {"start": lambda *a, **k: None})
worker_stub.should_emit_events = lambda *a, **k: False
sys.modules["prefect.events.worker"] = worker_stub

clients_stub = types.ModuleType("prefect.events.clients")
clients_stub.get_events_client = lambda *a, **k: None
clients_stub.get_events_subscriber = lambda *a, **k: None
clients_stub.AssertingEventsClient = type("AssertingEventsClient", (), {})
clients_stub.AssertingPassthroughEventsClient = type("AssertingPassthroughEventsClient", (), {})
clients_stub.PrefectCloudEventsClient = type("PrefectCloudEventsClient", (), {})
clients_stub.PrefectEventsClient = type("PrefectEventsClient", (), {})
sys.modules["prefect.events.clients"] = clients_stub

# Import Prefect safely
from prefect import flow, task, get_run_logger
from prefect.context import use_profile


# ------------------- TASK 1: Load Data -------------------
@task
def load_preprocessed_data():
    """Load preprocessed dataset (after data pipeline)."""
    logger = get_run_logger()
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_path = os.getenv("DATA_PATH", os.path.join(base_dir, "data", "customer_churn.csv"))
        df = pd.read_csv(data_path)
        logger.info(f"✅ Data loaded for ML. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


# ------------------- TASK 2: Prepare Features -------------------
@task
def prepare_features(df):
    """Clean dataset, encode categoricals, and split into train/test."""
    logger = get_run_logger()
    logger.info("🧩 Preparing features...")

    # Preserve customer IDs for later joining
    customer_ids = df["customerID"] if "customerID" in df.columns else None

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("🗑️ Dropped 'customerID' column.")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.info(f"🔠 Encoded {col}")

    if "Churn" not in df.columns:
        raise ValueError("❌ Target column 'Churn' not found.")
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if customer_ids is not None:
        customer_ids = customer_ids.iloc[y_test.index].reset_index(drop=True)

    logger.info(f"✅ Data split into train ({X_train.shape}) and test ({X_test.shape})")
    return X_train, X_test, y_train, y_test, customer_ids


# ------------------- TASK 3: Train Models -------------------
@task
def train_models(X_train, y_train):
    """Train multiple ML models."""
    logger = get_run_logger()
    logger.info("🚀 Training models...")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"🧠 Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"✅ {name} trained successfully!")

    return trained_models


# ------------------- TASK 4: Evaluate Models -------------------
@task
def evaluate_models(trained_models, X_test, y_test):
    """Evaluate trained models and save metrics."""
    logger = get_run_logger()
    results = []
    os.makedirs("src/plots", exist_ok=True)

    for name, model in trained_models.items():
        logger.info(f"📊 Evaluating {name}...")
        y_pred = model.predict(X_test)

        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0)
        }
        results.append({**metrics, "model": model})
        logger.info(f"✅ {name}: {metrics}")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"src/plots/{name.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.close()

    # Save metrics without model objects
    save_results = [{k: v for k, v in m.items() if k != "model"} for m in results]
    with open("src/plots/model_metrics.json", "w") as f:
        json.dump(save_results, f, indent=4)
    logger.info("📁 Saved model metrics to src/plots/model_metrics.json")

    best_model = max(results, key=lambda x: x["f1_score"])
    logger.info(f"🏆 Best model: {best_model['model_name']} | F1: {best_model['f1_score']:.4f}")
    return results, best_model


# ------------------- TASK 5: Predict Churn -------------------
@task
def predict_churn(best_model, X_test, y_test, customer_ids=None):
    """Predict which customers are likely to churn."""
    logger = get_run_logger()
    logger.info("🔮 Generating churn predictions...")

    model = best_model["model"]
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = np.zeros(len(y_pred))

    results_df = X_test.copy()
    if customer_ids is not None:
        results_df.insert(0, "customerID", customer_ids.values)

    results_df["Actual_Churn"] = y_test.values
    results_df["Predicted_Churn"] = y_pred
    results_df["Churn_Probability"] = y_prob

    os.makedirs("src/plots", exist_ok=True)
    results_path = "src/plots/predicted_churn.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"📁 Saved churn predictions to {results_path}")

    churned = results_df[results_df["Predicted_Churn"] == 1]
    logger.info(f"💔 Customers predicted to churn: {len(churned)} / {len(results_df)}")

    high_risk = results_df[results_df["Churn_Probability"] > 0.8]
    high_risk.to_csv("src/plots/high_risk_customers.csv", index=False)
    logger.info(f"⚠️ High-risk customers (>80% prob): {len(high_risk)}")

    # Plot top churn-risk customers
    top_risks = high_risk.sort_values("Churn_Probability", ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="Churn_Probability", y=top_risks["customerID"] if "customerID" in top_risks else top_risks.index,
        data=top_risks, orient="h", palette="Reds_r"
    )
    plt.title("Top 15 High-Risk Customers")
    plt.xlabel("Churn Probability")
    plt.ylabel("Customer ID" if "customerID" in top_risks else "Index")
    plt.tight_layout()
    plt.savefig("src/plots/top_churn_risks.png")
    plt.close()

    return results_df


# ------------------- MAIN FLOW -------------------
@flow(name="customer-churn-ml-pipeline")
def ml_pipeline_flow():
    """Main ML pipeline for Customer Churn Prediction."""
    logger = get_run_logger()
    logger.info("🚀 Starting Customer Churn ML Pipeline...")

    df = load_preprocessed_data()
    X_train, X_test, y_train, y_test, customer_ids = prepare_features(df)
    trained_models = train_models(X_train, y_train)
    results, best_model = evaluate_models(trained_models, X_test, y_test)
    predicted_df = predict_churn(best_model, X_test, y_test, customer_ids)

    logger.info("🎉 ML Pipeline executed successfully!")
    return results, best_model, predicted_df


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    with use_profile("ephemeral"):
        ml_pipeline_flow()
