import pytest
import pandas as pd
import pickle

MODEL_PATH = 'model.pkl'

@pytest.fixture
def loaded_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

def test_model_prediction(loaded_model):
    test_input = pd.DataFrame({
        "Driver_Age_Band": [2],
        "Time_of_day": [2],
        "Weather_Conditions": [2],
        "Speed": [30],
        "Driving_Experience": [5],
        "Driver_Score": [20]
    })

    prediction = loaded_model.predict(test_input)

    assert prediction is not None, "Model failed to make a prediction."
    assert len(prediction) == len(test_input), "Number of predictions does not match number of input samples."
