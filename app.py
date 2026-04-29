from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model = joblib.load("models/model_v2.pkl")


# 🔹 Home page (UI)
@app.route("/")
def home():
    return render_template("index.html")


# 🔹 Form-based prediction (from UI)
@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        # ✅ Extract values
        year = int(request.form["year"])
        month = int(request.form["month"])
        retail_sales = float(request.form["retail_sales"])
        retail_transfers = float(request.form["retail_transfers"])

        # ✅ Validation
        if month < 1 or month > 12:
            return render_template("index.html", prediction_text="❌ Month must be between 1 and 12")

        if year < 2000 or year > 2100:
            return render_template("index.html", prediction_text="❌ Enter a valid year (2000–2100)")

        if retail_sales < 0 or retail_transfers < 0:
            return render_template("index.html", prediction_text="❌ Values cannot be negative")

        # ✅ Create data (same as your original)
        data = {
            "year": year,
            "month": month,
            "retail_sales": retail_sales,
            "retail_transfers": retail_transfers
        }

        df = pd.DataFrame([data])
        df = df[["year", "month", "retail_sales", "retail_transfers"]]

        prediction = model.predict(df)

        return render_template(
            "index.html",
            prediction_text=f"Warehouse Sales : {round(float(prediction[0]), 2)}"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# 🔹 API prediction (for Postman / curl)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])
        df = df[["year", "month", "retail_sales", "retail_transfers"]]

        prediction = model.predict(df)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# 🔹 Run server
if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000)