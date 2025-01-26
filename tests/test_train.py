import pickle
from sklearn.datasets import load_iris

def test_model_output():
    # Load the saved model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load data
    data = load_iris()
    predictions = model.predict(data.data[:5])
    assert len(predictions) == 5

