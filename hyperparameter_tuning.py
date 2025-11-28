import mlflow
import mlflow.sklearn
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

# Set the local tracking URI (this points to a local folder for storing MLflow artifacts)
mlflow.set_tracking_uri("file:///C:/MLFlow-Project-ML/mlops-databricks-credit-default/mlruns")  # Use local directory

# Load the datasets
train_df = pd.read_csv('data/train_set.csv')
test_df = pd.read_csv('data/test_set.csv')

# Prepare features and target
target_column = 'default'
X_train = train_df.drop(columns=['ID', target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=['ID', target_column])
y_test = test_df[target_column]

# Initialize LightGBM model
model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42)

# Train the model with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    early_stopping_rounds=10,
    verbose=True
)

# Evaluate model performance
y_pred = model.predict(X_test)
auc_score = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {auc_score}")

# Log the model using MLflow
with mlflow.start_run():
    mlflow.log_param("model_type", "LightGBM with early stopping")
    mlflow.log_metric("AUC", auc_score)
    mlflow.sklearn.log_model(model, artifact_path="lightgbm_model")