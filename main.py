from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['Temperature', 'People', 'AC_Hours', 'Appliances', 'DayType']]
y = data['Electricity_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


@app.route('/')
def home():
    return render_template("index.html")


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

    return render_template("index.html", prediction_text=f"Estimated Electricity Consumption: {prediction} units")


if __name__ == "__main__":
    app.run(debug=True)