🚀 End-to-End MLOps Pipeline – Fraud Detection System
📌 Overview

This project implements a production-style MLOps pipeline for detecting fraudulent credit card transactions.
It demonstrates the complete lifecycle of a machine learning system, including data preprocessing, model training, experiment tracking, artifact management, and real-time inference via a REST API.

The focus of this project is operationalizing machine learning, not just building a model.

✨ Key Highlights

End-to-end ML pipeline (train → track → serve)

Experiment tracking with MLflow

Clear separation of code, data, and model artifacts

Real-time inference using FastAPI

Input validation with Pydantic

Health-check endpoint for monitoring

Interactive API documentation via Swagger (OpenAPI)

Docker-ready architecture for deployment

🏗️ Architecture Overview
Raw Data
   ↓
Data Preprocessing
   ↓
Model Training (Random Forest)
   ↓
MLflow Experiment Tracking
   ↓
Model Artifact (/models/model.pkl)
   ↓
FastAPI Inference Service

📂 Project Structure
ML OPS End-to-End ML pipeline/
│
├── api/
│   └── main.py            # FastAPI inference service
│
├── src/
│   ├── train.py           # Model training + MLflow logging
│   └── preprocess.py      # Data preprocessing
│
├── models/
│   └── model.pkl          # Trained model artifact
│
├── data/
│   └── raw/
│       └── creditcard.csv # Dataset
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── LICENSE
└── README.md

📊 Dataset

Dataset: Credit Card Fraud Detection

Transactions: ~284,000

Fraud cases: 492 (highly imbalanced)

Features: 30 numerical features (PCA-transformed)

Target: Class (0 = Normal, 1 = Fraud)

⚙️ Tech Stack

Language: Python 3.10

Machine Learning: Scikit-learn (Random Forest)

MLOps: MLflow

API: FastAPI, Uvicorn

Validation: Pydantic

Documentation: Swagger / OpenAPI

🚀 How to Run Locally
1️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python src/train.py


This will:

Train the model

Log metrics to MLflow

Save the model to models/model.pkl

4️⃣ Start the API
venv\Scripts\python.exe -m uvicorn api.main:app --reload

🌐 API Endpoints
🔹 Health Check
GET /health


Response:

{
  "status": "ok",
  "model_loaded": true
}

🔹 Fraud Prediction
POST /predict


Request:

{
  "features": [30 numerical values]
}


Response:

{
  "fraud_prediction": 0,
  "fraud_probability": 0.000137
}

📘 Swagger UI

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

📈 Model Evaluation

The following metrics are logged and tracked using MLflow:

ROC-AUC

Precision

Recall

Each training run is reproducible and versioned.

🧪 Validation & Monitoring

Input schema enforced using Pydantic

Feature length validation (expects exactly 30 features)

/health endpoint enables service monitoring and readiness checks

🔮 Future Enhancements

MLflow Model Registry (Staging → Production)

Automated retraining on data drift

Batch inference endpoint

Cloud deployment (AWS / GCP / Azure)

Kubernetes integration

📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

👤 Author

Harshavarthanan S
B.Tech – Artificial Intelligence & Data Science
