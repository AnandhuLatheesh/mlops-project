from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/model_v2.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        data = {
            "year": int(request.form["year"]),
            "month": int(request.form["month"]),
            "retail_sales": float(request.form["retail_sales"]),
            "retail_transfers": float(request.form["retail_transfers"])
        }

        # ✅ Validation
        if not (1 <= data["month"] <= 12):
            return render_template("index.html", prediction="Month must be between 1 and 12")

        df = pd.DataFrame([data])
        df = df[["year", "month", "retail_sales", "retail_transfers"]]

        prediction = round(model.predict(df)[0], 2)

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)