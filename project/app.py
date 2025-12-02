from flask import Flask, render_template, request
import numpy as np
import onnxruntime as ort
import joblib
import os

app = Flask(__name__)

# RUTAS ABSOLUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar ONNX y SCALERS
session = ort.InferenceSession(
    os.path.join(BASE_DIR, "model.onnx"),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

scaler_X = joblib.load(os.path.join(BASE_DIR, "scaler_X.pkl"))
scaler_Y = joblib.load(os.path.join(BASE_DIR, "scaler_Y.pkl"))

MODEL_MAE = 0.0341
escaled_MAE = scaler_Y.inverse_transform([[MODEL_MAE]])[0][0]


# RUTA PRINCIPAL
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
        heart_rate=None,
        temperature=None
    )

# PREDICCIÓN
@app.route("/predict", methods=["POST"])
def predict():

    # Obtener datos del formulario
    sex = request.form["sex"]
    sex_bin = float(sex == "Male")

    age = float(request.form["age"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])
    duration = float(request.form["duration"])
    heart_rate = float(request.form["heart_rate"])
    temperature = float(request.form["temperature"])

    # Vector features
    X = np.array([[sex_bin, age, height, weight, duration, heart_rate, temperature]],
                 dtype=np.float32)

    # Escalar entrada
    X_scaled = scaler_X.transform(X).astype(np.float32)

    # Ejecutar ONNX
    result = session.run([output_name], {input_name: X_scaled})
    pred_scaled = float(result[0])     # convertimos a float seguro

    # Desescalar a calorías reales
    y_pred = scaler_Y.inverse_transform([[pred_scaled]])[0][0]

    # Calcular confianza basada en MAE
    confidence = max(0, (1 - MODEL_MAE) * 100)
    confidence = round(confidence, 2)

    # Regresar predicción y mantener campos llenos
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
        temperature=temperature
    )

# RUN LOCAL
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
