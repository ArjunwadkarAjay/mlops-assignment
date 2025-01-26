import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import pandas as pd

# Load dataset
# data = load_iris()

# Load from a CSV file
# Load data from CSV file
data = pd.read_csv('data/dataset_iris.csv')

# Assuming the last column is the target variable (label) and all others are features
X = data.iloc[:, :-1].values  # Select all rows and all columns except the last one
y = data.iloc[:, -1].values   # Select the last column as the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start an MLflow run
with mlflow.start_run():
    # Define and train the model

    # Define hyperparameters
    n_estimators = 100
    max_depth = 3

    # Train the model
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Infer the signature
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train[:5]  # Use a small slice of the input data as an example

    # Log the model with signature and input example
    mlflow.sklearn.log_model(
        model,
        "random_forest_model",
        signature=signature,
        input_example=input_example,
    )

print(f"Model Accuracy: {accuracy:.4f}")
print("Experiment logged to MLflow.")