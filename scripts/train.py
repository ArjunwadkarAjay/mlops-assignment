import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import pandas as pd
import hashlib

# Calculate dataset checksum (hash)
dataset_path = 'data/dataset_iris.csv'
with open(dataset_path, 'rb') as f:
    dataset_hash = hashlib.md5(f.read()).hexdigest()

# Load data from a CSV file
data = pd.read_csv('data/dataset_iris.csv')

# Assuming the last column is the target variable (label)
# and all others are features

# Select all rows and all columns except the last one
X = data.iloc[:, :-1].values

# Select the last column as the target variable
y = data.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start the main MLflow run
with mlflow.start_run():
    # Log dataset details in parent run
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("dataset_hash", dataset_hash)
    mlflow.log_param("dataset_version", "v1.0")

    # Define hyperparameter grids
    n_estimators = [100, 150, 200]
    max_depth = [3, 4, 5]

    for i in range(len(n_estimators)):
        # Start a nested MLflow run for each hyperparameter combination
        with mlflow.start_run(nested=True):
            # Train the model
            model = RandomForestClassifier(
                n_estimators=n_estimators[i],
                max_depth=max_depth[i],
                random_state=42
            )
            model.fit(X_train, y_train)

            # Evaluate the model
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Log parameters and metrics for this model
            mlflow.log_param("n_estimators", n_estimators[i])
            mlflow.log_param("max_depth", max_depth[i])
            mlflow.log_metric("accuracy", accuracy)

            # Infer the signature and input example
            signature = infer_signature(X_train, model.predict(X_train))

            # Use a small slice of the input data as an example
            input_example = X_train[:5]

            # Log the model with signature and input example
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                signature=signature,
                input_example=input_example,
            )

            print(f"Model's n_estimators={n_estimators[i]}")
            print(f"Model's max_depth={max_depth[i]}")
            print(f"Accuracy: {accuracy:.4f}")

print("Experiments logged to MLflow.")
