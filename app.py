from flask import Flask, request, jsonify
import mlflow.sklearn

# Load the model from MLflow for GridSearchCV approach
model_uri = "mlruns/0/ee30eaa72f7842099429a76e82910ad9/artifacts/random_forest_best_model"
model = mlflow.sklearn.load_model(model_uri)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.json
        input_data = data['input']

        # Predict using the model
        prediction = model.predict([input_data])
        response = {'prediction': prediction[0]}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
