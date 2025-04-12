import numpy as np
import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest.pkl")
autoencoder = load_model("autoencoder.h5", compile=False)
threshold = joblib.load("autoencoder_threshold.pkl")

def preprocess_input(user_input):
    arr = np.array(user_input).reshape(1, -1)
    return scaler.transform(arr)

def predict_combined(scaled_input):
    rf_pred = rf_model.predict(scaled_input)[0]
    recon = autoencoder.predict(scaled_input)
    recon_error = np.mean(np.square(scaled_input - recon))

    debug_info = {
        "rf_pred": rf_pred,
        "recon_error": recon_error,
        "threshold": threshold
    }

    if recon_error > threshold or rf_pred == 1:
        result = "Attack Detected"
    else:
        result = "Normal Traffic"

    return result, debug_info
