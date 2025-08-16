from flask import Flask, request, render_template
import numpy as np
import pickle

#app = Flask(__name__)
app = Flask(__name__, template_folder="template")

# Load models
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


aqi_model = load_model("models/air_reg.pkl")
health_model = load_model("models/air_cls.pkl")
water_model = load_model("models/water_quality_model.pkl")


# ----- ROUTES -----


@app.route("/report")
def intro_report():
    return render_template("report.html")


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/air_form")
def air_form():
    return render_template("home1.html")  # Air quality form


@app.route("/water_form")
def water_form():
    return render_template("home2.html")  # Water quality form


# ----- AIR QUALITY WORKFLOW -----


@app.route("/predict", methods=["POST"])
def predict_air():
    try:
        inputs = [float(request.form[x]) for x in ["pm25", "pm10", "o3", "so2", "no2"]]
        aqi = aqi_model.predict([inputs])[0]
        health_inputs = inputs + [aqi]
        health_class = health_model.predict([health_inputs])[0]
        health_label = "Healthy" if health_class == 0 else "Unhealthy"

        return render_template(
            "result1.html", aqi=round(aqi, 2), health_class=health_label
        )

    except Exception as e:
        return f"Error in air quality prediction: {str(e)}"


# ----- WATER QUALITY WORKFLOW -----


# ✅ Define classification logic, globally
def classify_potability(wqi):
    if wqi > 70:
        return "Potable"
    elif wqi < 50:
        return "Non-potable"
    else:
        return "Possibly Potable"


@app.route("/predict_water", methods=["POST"])
def predict_water():
    try:

        def safe_float(field):
            return float(request.form.get(field, 0) or 0)

        feature_list = [
            "pH",
            "CO3",
            "HCO3",
            "Cl",
            "SO4",
            "NO3",
            "TH",
            "Ca",
            "Mg",
            "Na",
            "K",
            "F",
            "TDS",
        ]

        values = [safe_float(feat) for feat in feature_list]

        # ✅ Predict WQI
        predicted_wqi = water_model.predict([values])[0]

        # ✅ Apply threshold to classify potability
        potability = classify_potability(predicted_wqi)

        # ✅ Pass to result2.html
        return render_template(
            "result2.html", wqi=round(predicted_wqi, 2), potability=potability
        )

    except Exception as e:
        return f"Error in water quality prediction: {str(e)}"


# ----- RUN APP -----

if __name__ == "__main__":
    app.run(debug=True)
