# ============================================================
# src/api_client.py
# Customer Churn Project - API Client (Offline Compatible)
# Works even when no Prefect server or orchestration is running
# ============================================================

import os
import json
import requests
import logging

# ------------------- Configuration -------------------
USE_PREFECT_API = False   # Set True only if Prefect server is running
PREFECT_API_URL = "http://localhost:4200/api"
os.makedirs("plots", exist_ok=True)

# ------------------- Logging Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("plots/api_monitoring.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ------------------- Helper Function -------------------
def query_prefect_api(query, variables=None):
    """Query Prefect GraphQL API (or offline stub)."""
    if not USE_PREFECT_API:
        logger.warning("⚙️  Offline mode active — skipping actual API call.")
        return {"data": {"message": "Offline mode: Prefect API not queried."}}

    try:
        response = requests.post(
            f"{PREFECT_API_URL}/graphql",
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API Error: {e}")
        return {"error": str(e)}

# ------------------- API Query Functions -------------------
def get_flows_api():
    """Get list of all flows."""
    logger.info("Fetching Flows...")
    query = """
    query {
        flows {
            id
            name
            created
            updated
            tags
        }
    }
    """
    return query_prefect_api(query)

def get_deployments_api():
    """Get list of all deployments."""
    logger.info("Fetching Deployments...")
    query = """
    query {
        deployments {
            id
            name
            flow_id
            schedule {
                cron
            }
            created
            updated
        }
    }
    """
    return query_prefect_api(query)

def get_flow_runs_api():
    """Get list of flow runs."""
    logger.info("Fetching Flow Runs...")
    query = """
    query {
        flow_runs(limit: 5, order_by: {start_time: DESC}) {
            id
            name
            state_type
            start_time
            end_time
            total_run_time
        }
    }
    """
    return query_prefect_api(query)

def get_task_runs_api():
    """Get list of task runs."""
    logger.info("Fetching Task Runs...")
    query = """
    query {
        task_runs(limit: 5, order_by: {start_time: DESC}) {
            id
            name
            task_key
            state_type
            start_time
            end_time
        }
    }
    """
    return query_prefect_api(query)

# ------------------- Main Runner -------------------
def run_api_monitoring():
    """Run all API checks and save results."""
    logger.info("=" * 60)
    logger.info("STARTING API MONITORING DEMO")

    results = {
        "flows": get_flows_api(),
        "deployments": get_deployments_api(),
        "flow_runs": get_flow_runs_api(),
        "task_runs": get_task_runs_api(),
    }

    output_path = "plots/api_responses.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"API responses saved at {output_path}")
    logger.info("API MONITORING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    return results

# ------------------- Entry Point -------------------
if __name__ == "__main__":
    run_api_monitoring()
