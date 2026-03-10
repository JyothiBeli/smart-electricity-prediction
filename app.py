import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("AI Based Smart Electricity Consumption Predictor")

# Load dataset
data = pd.read_csv("dataset.csv")

X = data[['Temperature', 'People', 'AC_Hours', 'Appliances', 'DayType']]
y = data['Electricity_Consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.header("Enter Details")

temp = st.number_input("Temperature")
people = st.number_input("Number of People")
ac = st.number_input("AC Hours")
appliances = st.number_input("Number of Appliances")
day = st.selectbox("Day Type (0=Weekday, 1=Weekend)", [0,1])

if st.button("Predict"):
    prediction = model.predict([[temp, people, ac, appliances, day]])
    st.success(f"Estimated Electricity Consumption: {round(prediction[0],2)} units")