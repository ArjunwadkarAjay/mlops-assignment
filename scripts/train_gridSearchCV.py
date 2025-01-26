import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load dataset
data = pd.read_csv('data/dataset_iris.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Define model and hyperparameters for tuning
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Perform hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)  # Fit the model here

# Best parameters and accuracy
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log results to MLflow
with mlflow.start_run():
    # Infer model signature and input example
    signature = infer_signature(X_train, best_model.predict(X_train[:5]))
    input_example = X_train[0].tolist()

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        best_model,
        "random_forest_best_model",
        signature=signature,
        input_example={"input": input_example}
    )

# Save best parameters
print("Best Parameters:", best_params)
print(f"Test Accuracy: {accuracy:.4f}")
