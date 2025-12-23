import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from preprocess import preprocess

print("STEP 1: Script started")

DATA_PATH = "data/raw/creditcard.csv"
print("STEP 2: Loading data from", DATA_PATH)

X_train, X_test, y_train, y_test = preprocess(DATA_PATH)
print("STEP 3: Data loaded successfully")

mlflow.set_experiment("Credit-Card-Fraud-MLOps")

with mlflow.start_run():
    print("STEP 4: MLflow run started")

    model = RandomForestClassifier(
        n_estimators=50,   # reduced for speed
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    print("STEP 5: Model trained")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("STEP 6: Predictions generated")

    mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))
    mlflow.log_metric("precision", precision_score(y_test, preds))
    mlflow.log_metric("recall", recall_score(y_test, preds))

    print("STEP 7: Metrics logged")

    # FORCE MODEL SAVE
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, MODEL_PATH)

    print("STEP 8: Model saved to", MODEL_PATH)

    mlflow.sklearn.log_model(model, "model")

print("STEP 9: Script finished successfully")
