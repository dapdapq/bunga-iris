import joblib
import numpy as np

MODEL_PATH = "knn_model.sav"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def predict_species(model, features):
    """
    features: list [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
    """
    x = np.array(features).reshape(1, -1)
    prediction = model.predict(x)
    return prediction[0]
