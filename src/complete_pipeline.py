# ============================================================
# src/complete_pipeline.py
# Complete Customer Churn Prediction Pipeline (Fully Offline)
# Combines data_pipeline + ml_pipeline with Prefect stubs
# ============================================================

import os
import sys
import types
import json

# ------------------- Disable Prefect Cloud/Server Calls -------------------
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

# ------------------- Safe Prefect Import -------------------
from prefect import flow, get_run_logger
from prefect.context import use_profile

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import data_pipeline_flow
from src.ml_pipeline import ml_pipeline_flow


@flow(name="complete-customer-churn-pipeline")
def complete_pipeline():
    """Complete pipeline combining data and ML steps"""
    logger = get_run_logger()
    logger.info("🎯 STARTING COMPLETE CUSTOMER CHURN PIPELINE")
    logger.info("=" * 60)

    try:
        # --------------------------
        # 📊 PHASE 1: Data Pipeline
        # --------------------------
        logger.info("📊 PHASE 1: Running Data Pipeline...")
        processed_data, feature_importance = data_pipeline_flow()
        logger.info("✅ Data Pipeline completed successfully!")

        # --------------------------
        # 🤖 PHASE 2: ML Pipeline
        # --------------------------
        logger.info("🤖 PHASE 2: Running ML Pipeline...")
        ml_results, best_model, predicted_df = ml_pipeline_flow()
        logger.info("✅ ML Pipeline completed successfully!")

        # --------------------------
        # 🧹 CLEANUP: Remove model objects before saving
        # --------------------------
        safe_results = []
        for m in ml_results:
            safe_results.append({k: v for k, v in m.items() if k != "model"})

        safe_best_model = {k: v for k, v in best_model.items() if k != "model"}

        # --------------------------
        # 📈 Save Results
        # --------------------------
        logger.info(f"🏆 Best Model: {best_model['model_name']} | F1: {best_model['f1_score']:.4f}")

        metrics_path = "src/plots/model_metrics_final.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "results": safe_results,
                    "best_model": safe_best_model,
                    "predicted_churn_file": "src/plots/predicted_churn.csv",
                },
                f,
                indent=4
            )

        logger.info(f"📁 Model metrics saved at: {metrics_path}")
        logger.info(f"💾 Predicted churn file: src/plots/predicted_churn.csv")
        logger.info("=" * 60)
        logger.info("✅ COMPLETE PIPELINE EXECUTED SUCCESSFULLY!")

        return {"ml_results": safe_results, "best_model": safe_best_model, "predicted": predicted_df}

    except Exception as e:
        logger.error(f"❌ Pipeline failed due to error: {e}")
        raise


if __name__ == "__main__":
    with use_profile("ephemeral"):
        complete_pipeline()
