Predictive Customer Churn Analysis (Cloud-Native MLOps Project)
Overview

This project demonstrates a cloud-native, end-to-end MLOps pipeline for predicting customer churn using real-world data.
It integrates DataOps, MLOps, and API-driven orchestration using modern cloud tools such as Prefect, Docker, and Kubernetes.

Project Architecture
Data Source (Kaggle CSV)
       ↓
Data Pipeline (Prefect Flow)
       ↓
Preprocessing + EDA + Feature Engineering
       ↓
ML Pipeline (Logistic Regression, Random Forest, XGBoost)
       ↓
Model Evaluation + Metrics Logging
       ↓
FastAPI Layer (Model metrics + status)
       ↓
Docker Container
       ↓
Kubernetes CronJob (Scheduled execution)

Objectives

Automate data ingestion, preprocessing, and EDA.
Train and evaluate multiple ML models for churn prediction.
Expose model performance metrics through an API.
Containerize and schedule the workflow for cloud deployment.

Tech Stack
Category	Tools / Libraries
Languages	Python
Data & ML	Pandas, NumPy, Scikit-learn, XGBoost
Visualization	Matplotlib, Seaborn
Workflow Orchestration	Prefect
API Framework	FastAPI
Containerization	Docker
Orchestration	Kubernetes CronJob
Storage / Logs	JSON, CSV, Prefect Logs
📁 Project Structure
📦 predictive-churn-analysis
 ┣ 📂 data/                  # Dataset (customer_churn.csv)
 ┣ 📂 src/
 ┃ ┣ 📜 data_pipeline.py     # Prefect-based data flow (EDA, preprocessing)
 ┃ ┣ 📜 ml_pipeline.py       # Model training and evaluation
 ┃ ┣ 📜 complete_pipeline.py # Combined flow (Data + ML)
 ┃ ┣ 📜 api_client.py        # FastAPI mock API
 ┃ ┣ 📜 verify_project.py    # Verification and validation script
 ┃ ┗ 📂 plots/               # Model metrics, confusion matrices, logs
 ┣ 📜 Dockerfile             # Container configuration
 ┣ 📜 churn-cronjob.yaml     # Kubernetes CronJob manifest
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 README.md              # Project documentation
 ┗ 📜 entrypoint.sh          # Container entry script

Model Performance
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	91.24%	1.00	0.005	0.01
Random Forest	94.08%	0.95	0.34	0.50
XGBoost	93.94%	0.92	0.33	0.49

Best Model: Random Forest (Accuracy 94.08%, F1 Score 0.50)

Deployment

Containerization: Docker image built with all dependencies preinstalled.
Orchestration: Kubernetes CronJob runs pipeline daily at 2 AM.
Automation: Prefect manages data + ML flows with logs and monitoring.
Offline Mode: Fully functional without Prefect Cloud dependencies.

API Access

The system exposes key application metrics through a REST API built with FastAPI.

Example Endpoint:

GET /app-info


Sample Response:

{
  "status": "running",
  "last_run": "2025-10-20 22:30",
  "accuracy": 0.94,
  "precision": 0.95,
  "recall": 0.34,
  "pipeline_status": "completed"
}

Results

Automated churn analysis pipeline from ingestion to prediction.
Random Forest identified at-risk customers with 50% recall accuracy.
Metrics, logs, and predictions stored in /src/plots/ for monitoring.
Reproducible, cloud-ready, and deployment-optimized system.

Future Enhancements

Handle class imbalance using SMOTE or cost-sensitive training.
Integrate Prometheus + Grafana for live monitoring.

Deploy FastAPI as a standalone microservice.

Add CI/CD pipeline for automated retraining.
