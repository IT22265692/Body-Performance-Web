from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained pipeline model
model = joblib.load("xgb_model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        age = float(data["age"])
        height = float(data["height"])
        weight = float(data["weight"])
        duration = float(data["duration"])
        heart_rate = float(data["heart_rate"])
        body_temp = float(data["body_temp"])
        sex = data["sex"]

        # 🔥 Calculate extra features
        bmi = weight / ((height / 100) ** 2)
        max_hr = 220 - age

        input_data = pd.DataFrame([{
            "Sex": sex,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
        }])

        prediction = model.predict(input_data)

        return jsonify({
            "calories_burned": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)