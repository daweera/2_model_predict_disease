from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)


def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


current_directory = os.getcwd()
diabetes_model_path = os.path.join(
    current_directory,
    "Multiple_Disease_Prediction-main__",
    "saved_models",
    "diabetes.pkl",
)
heart_model_path = os.path.join(
    current_directory, "Multiple_Disease_Prediction-main__", "saved_models", "heart.pkl"
)


diabetes_model = load_model(diabetes_model_path)
heart_model = load_model(heart_model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.json
        features = np.array(
            [
                data["pregnancies"],
                data["glucose"],
                data["bloodpressure"],
                data["skinthickness"],
                data["insulin"],
                data["bmi"],
                data["dpf"],
                data["age"],
                # ฟีเจอร์เพิ่มเติมตามที่โมเดล
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ).reshape(1, -1)
        prediction = diabetes_model.predict(features)[0]
        result = (
            "มีความเสี่ยงที่จะเป็นโรคเบาหวาน"
            if prediction == 1
            else "ไม่มีความเสี่ยงที่จะเป็นโรคเบาหวาน"
        )
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.json
        features = [
            data["age"],
            data["sex"],
            data["cp"],
            data["trestbps"],
            data["chol"],
            data["fbs"],
            data["restecg"],
            data["thalach"],
            data["exang"],
            data["oldpeak"],
            data["slope"],
            data["ca"],
            data["thal"],
        ]
        prediction = heart_model.predict([features])[0]
        result = (
            "มีความเสี่ยงที่จะเป็นโรคหัวใจ" if prediction == 1 else "ไม่มีความเสี่ยงที่จะเป็นโรคหัวใจ"
        )
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
