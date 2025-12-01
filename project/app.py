from flask import Flask, render_template, request
import numpy as np
import torch
from model_loader import load_model
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = load_model(os.path.join(BASE_DIR, "best_model_regression.pth"))

scaler_X = joblib.load(os.path.join(BASE_DIR, "scaler_X.pkl"))
scaler_Y = joblib.load(os.path.join(BASE_DIR, "scaler_y.pkl"))

@app.route("/")
def index():
    return render_template(
        "index.html",
        prediction=None,
        confidence=None,
        sex=None,
        age=None,
        height=None,
        weight=None,
        duration=None,
        heartbeat=None,
        temperature=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    # Obtener datos del formulario
    User_ID = 0.0
    sex = request.form["sex"]
    sex_bin = float(sex == "Male")
    age = float(request.form["age"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    duration = float(request.form["duration"])
    heart_rate = float(request.form["heart_rate"])
    temperature = float(request.form["temperature"])

    # Crear vector numpy (orden debe ser igual que en entrenamiento)
    X = np.array([[User_ID, sex_bin, age, height, weight, duration, heart_rate, temperature]], dtype=np.float32)

    # Escalar igual que en entrenamiento
    X_scaled = scaler_X.transform(X)

    # Convertir a tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Predicción con PyTorch
    with torch.no_grad():
        prediction = model(X_tensor).numpy()
    # Desescalar (invertir)
    y_pred = scaler_Y.inverse_transform(prediction.reshape(-1, 1))[0][0]

    MODEL_MAE = 0.0150  # tu MAE

    # Confianza basada en MAE normalizado (0–100%)
    confidence = max(0, (1 - MODEL_MAE) * 100)
    confidence = round(confidence, 2)

    return render_template(
        "index.html",
        prediction=round(y_pred, 2),
        confidence=confidence,
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        duration=duration,
        heart_rate=heart_rate,
        temperature=temperature)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)