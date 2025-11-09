import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

mlflow.set_tracking_uri('http://mlflow:'+ os.getenv('MLFLOW_PORT', '8000'))
mlflow.set_experiment("diabetes_rf_experiment")

class RandomForestModel:
    def __init__(self, params):
        self.model = RandomForestRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def create_new_model():
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 3, "max_features": 3}
        model = RandomForestModel(params)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(sk_model=model, name="RandomForestModel", input_example=X_train[:5], registered_model_name="RandomForestModel")
        mlflow.log_params(params)
        mlflow.log_metric("squared_error", mean_squared_error(y_test, model.predict(X_test)))
    return True