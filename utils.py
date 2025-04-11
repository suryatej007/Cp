import numpy as np
import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest.pkl")
autoencoder = load_model("autoencoder.h5")

THRESHOLD = 0.01  # tune this based on validation loss

def preprocess_input(user_input):
    data = np.array(user_input).reshape(1, -1)
    scaled = scaler.transform(data)
    return scaled

def predict_combined(scaled_input):
    rf_pred = rf_model.predict(scaled_input)[0]
    reconstructed = autoencoder.predict(scaled_input)
    ae_error = np.mean(np.square(scaled_input - reconstructed))

    if rf_pred == 1 or ae_error > THRESHOLD:
        return "Attack Detected"
    else:
        return "Normal Traffic"