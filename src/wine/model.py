import joblib
import json
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def train(train_data_file_path: Path,
          target_column: str, alpha: float, l1_ratio: float, model_path: Path):
    
    train_data = pd.read_csv(train_data_file_path)
    train_X = train_data.drop([target_column], axis=1)
    train_y = train_data[target_column]
     
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(train_X, train_y)
    # Save model
    joblib.dump(model, model_path)

def evaluate(model_path, test_data_file_path, target_column,
             metrics_file_path):
    # Predict on test set
    test_data = pd.read_csv(test_data_file_path)
    test_X = test_data.drop([target_column], axis=1)
    test_y = test_data[target_column]
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict(test_X)
    # Calculate metrics
    rmse = root_mean_squared_error(y_true=test_y, y_pred=y_pred)
    mae = mean_absolute_error(y_true=test_y, y_pred=y_pred)
    r2 = r2_score(y_true=test_y, y_pred=y_pred)
    # Save metrics to a file
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(metrics, metrics_file)

def track_with_mlflow(uri, experiment_name, params, metrics):
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=uri)
    # Create a new MLflow Experiment
    mlflow.set_experiment(experiment_name=experiment_name)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metrics(metrics)