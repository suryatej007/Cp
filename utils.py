import numpy as np
import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest.pkl")
autoencoder = load_model("autoencoder.h5", compile=False)
threshold = joblib.load("autoencoder_threshold.pkl")

recon = autoencoder.predict(X_scaled)
errors = np.mean(np.square(X_scaled - recon), axis=1)
threshold = np.percentile(errors, 95)
joblib.dump(threshold, "autoencoder_threshold.pkl")


def preprocess_input(user_input):
    arr = np.array(user_input).reshape(1, -1)
    return scaler.transform(arr)



def predict_combined(scaled_input):
    rf_pred = rf_model.predict(scaled_input)[0]
    recon = autoencoder.predict(scaled_input)
    error = np.mean(np.square(scaled_input - recon))
    print("🟡 RF Prediction:", rf_pred)
    print("🟡 AE Error:", error, "| Threshold:", threshold)

    if rf_pred == 1 or error > threshold:
        return "Attack Detected"
    else:
        return "Normal Traffic"

