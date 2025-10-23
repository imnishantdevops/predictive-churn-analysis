# src/data_pipeline.py
# Customer Churn Prediction - Data Pipeline (Fully Offline Mode)
# Works without any Prefect server, cloud, or event system.
# ============================================================

import os
import sys
import types

# ------------------- Disable Prefect Network Calls -------------------
os.environ["PREFECT_API_URL"] = ""
os.environ["PREFECT_API_ENABLE"] = "false"
os.environ["PREFECT_TEST_MODE"] = "true"
os.environ["PREFECT_LOCAL_MODE"] = "true"
os.environ["PREFECT_LOGGING_LEVEL"] = "INFO"
os.environ["PREFECT_ANALYTICS_ENABLED"] = "false"
os.environ["PREFECT_EXPERIMENTAL_ENABLE_EVENTS_CLIENT"] = "false"
os.environ["PREFECT_EXPERIMENTAL_ENABLE_EVENTS_WORKER"] = "false"

# ------------------- Stub Prefect Event Modules -------------------
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

# ------------------- Safe Import for Prefect -------------------
from prefect import flow, task, get_run_logger
from prefect.context import use_profile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ------------------- TASK 1: Load Data -------------------
@task
def load_data():
    """Load the Telco Customer Churn dataset"""
    logger = get_run_logger()
    try:
        # Use environment variable if available (Docker-safe)
        data_path = os.getenv("DATA_PATH", "data/customer_churn.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from: {data_path} | Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


# ------------------- TASK 2: Preprocess Data -------------------
@task
def preprocess_data(df):
    """Clean and preprocess the dataset"""
    logger = get_run_logger()
    logger.info("🧹 Starting preprocessing...")

    df.columns = df.columns.str.strip()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total = df["TotalCharges"].isnull().sum()
    if missing_total > 0:
        logger.info(f"Found {missing_total} missing TotalCharges values. Filling with median.")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"Encoded column: {col}")
        except Exception as e:
            logger.warning(f"Could not encode {col}: {e}")

    # Scale numeric features
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    logger.info("Preprocessing completed successfully.")
    logger.info(f"Data shape after preprocessing: {df.shape}")
    return df


# ------------------- TASK 3: EDA -------------------
@task
def perform_eda(df):
    """Perform exploratory data analysis"""
    logger = get_run_logger()
    os.makedirs("plots", exist_ok=True)
    logger.info("Starting EDA...")

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()

    if "Churn" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.countplot(x="Churn", data=df)
        plt.title("Churn Distribution (0=No, 1=Yes)")
        plt.tight_layout()
        plt.savefig("plots/churn_distribution.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="tenure", y="MonthlyCharges", hue="Churn", data=df, alpha=0.6)
        plt.title("Monthly Charges vs Tenure (by Churn)")
        plt.tight_layout()
        plt.savefig("plots/monthly_vs_tenure.png")
        plt.close()

        churn_corr = corr["Churn"].abs().sort_values(ascending=False)
        logger.info(f"Top correlated features with Churn:\n{churn_corr.head(10)}")
    else:
        logger.warning("'Churn' column not found, skipping correlation ranking.")
        churn_corr = pd.Series(dtype=float)

    logger.info("EDA completed successfully. Plots saved in 'plots' directory.")
    return churn_corr


# ------------------- MAIN FLOW -------------------
@flow(name="customer-churn-data-pipeline")
def data_pipeline_flow():
    """Main data pipeline for Customer Churn Prediction"""
    logger = get_run_logger()
    logger.info("Starting Customer Churn Data Pipeline...")

    df = load_data()
    processed_df = preprocess_data(df)
    feature_importance = perform_eda(processed_df)

    logger.info("Data Pipeline executed successfully!")
    return processed_df, feature_importance


# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    with use_profile("ephemeral"):
        data_pipeline_flow()
