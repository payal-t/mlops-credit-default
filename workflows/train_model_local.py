"""
Local training script for credit default prediction
- Uses CSV data instead of Databricks tables
- No Databricks authentication needed
- Logs metrics, parameters, and model to local MLflow
"""

import argparse
import os
import sys
import pandas as pd
from loguru import logger

import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from credit_default.utils import load_config, setup_logging

# Setup logging
setup_logging(log_file="")

try:
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, type=str, help="Root path of the project")
    parser.add_argument("--git_sha", required=True, type=str, help="Git commit SHA")
    parser.add_argument("--job_run_id", required=True, type=str, help="Job run ID")
    args = parser.parse_args()

    root_path = args.root_path
    git_sha = args.git_sha
    job_run_id = args.job_run_id
    logger.debug(f"Git SHA: {git_sha}")
    logger.debug(f"Job Run ID: {job_run_id}")
    logger.info("Parsed arguments successfully.")

    # Load configuration
    config_path = os.path.join(root_path, "project_config.yml")
    config = load_config(config_path)
    logger.info("Configuration loaded successfully.")

    # Load data locally
    train_csv_path = os.path.join(root_path, "data", "train_set.csv")
    test_csv_path = os.path.join(root_path, "data", "test_set.csv")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) datasets.")

    # Extract feature columns and target
    columns = config.features.clean
    columns_wo_id = [c for c in columns if c != "Id"]
    target = config.target[0].new_name
    features_robust = config.features.robust
    parameters = config.parameters

    X_train = train_df[columns_wo_id]
    y_train = train_df[target]
    X_test = test_df[columns_wo_id]
    y_test = test_df[target]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[("robust_scaler", RobustScaler(), features_robust)],
        remainder="passthrough"
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**parameters))]
    )

    # Local MLflow setup
    mlruns_path = os.path.join(root_path, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment("credit-default-local")
    logger.info(f"MLflow tracking URI set to local folder: {mlruns_path}")

    # Train and log model
    with mlflow.start_run(tags={"git_sha": git_sha, "job_run_id": job_run_id}):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        auc_test = roc_auc_score(y_test, y_pred)
        logger.info(f"Test AUC: {auc_test}")

        # Log params, metrics, and model
        mlflow.log_param("model_type", "LightGBM with preprocessing")
        mlflow.log_params(parameters)
        mlflow.log_metric("AUC", auc_test)
        mlflow.sklearn.log_model(pipeline, artifact_path="lightgbm-pipeline-model")

except Exception as e:
    logger.error(f"An error occurred: {e}")
    sys.exit(1)
