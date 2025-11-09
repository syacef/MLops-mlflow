import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv
from model import create_new_model
import random
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_experiment("diabetes_rf_experiment")
mlflow.set_tracking_uri(f"http://mlflow:{os.getenv('MLFLOW_PORT', '8000')}")

canary_RELEASE = False

app = FastAPI()
@app.post("/predict")
def predict(data: dict):
    client = MlflowClient()
    versions = client.search_model_versions("name='RandomForestModel'")
    if len(versions) >= 2 and (random.random() < float(os.getenv("PROBABILITY_OLD_MODEL", "0.5"))) and canary_RELEASE:
        sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        prev_version = sorted_versions[1].version
        model = mlflow.sklearn.load_model(f"models:/RandomForestModel/{prev_version}")
    else:
        model = mlflow.sklearn.load_model("models:/RandomForestModel/latest")
    predictions = model.predict(pd.DataFrame([data]))
    return {"predictions": predictions.tolist()}

@app.post("/update-model")
def update_model(data: dict):
    model = mlflow.sklearn.load_model("models:/RandomForestModel/latest")
    X = pd.DataFrame(data, index=[0])
    y = X["target"]
    X.drop(columns=["target"], inplace=True)
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "RandomForestModel", registered_model_name="RandomForestModel")
    return {"status": "model updated"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/create-model")
def create_model():
    create_new_model()
    return {"status": "model created"}

@app.post("/accept-next-model")
def accept_next_model():
    global canary_RELEASE
    canary_RELEASE = True
    return {"status": "canary release enabled"}

if __name__ == "__main__":
    print(f"Starting app on port {os.getenv('SERVER_PORT', '8080')}, mlflow -> {os.getenv('MLFLOW_PORT', '5000')}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", 8080)))
