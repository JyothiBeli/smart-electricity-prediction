from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

# Features and target
X = data[['Temperature', 'People', 'AC_Hours', 'Appliances', 'DayType']]
y = data['Electricity_Consumption']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Home page
@app.route('/')
def home():
    return render_template("index.html")


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    people = int(request.form['people'])
    ac = float(request.form['ac_hours'])
    appliances = int(request.form['appliances'])
    day = int(request.form['day_type'])

    new_data = [[temp, people, ac, appliances, day]]
    result = model.predict(new_data)

    prediction = round(result[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Estimated Electricity Consumption: {prediction} units"
    )


# Run the app (important for Railway deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)