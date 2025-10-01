import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_experiment("diabetes_rf_experiment")
mlflow.set_tracking_uri(f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}")

app = FastAPI()
@app.post("/predict")
def predict(data: dict):
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

#@app.get("/create-model")
#def create_model():
#    create_new_model()
#    return {"status": "model created"}

if __name__ == "__main__":
    print(f"Starting app on port {os.getenv('SERVER_PORT', '8080')}, mlflow -> {os.getenv('MLFLOW_PORT', '5000')}")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("SERVER_PORT", 8080)))
