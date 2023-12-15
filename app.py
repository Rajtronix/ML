import pandas as pd
import joblib
import streamlit as st

# Load the saved model
loaded_model = joblib.load('temperature_humidity_prediction_model.pkl')

# Function to predict next hour values
def predict_next_hour(user_input_temperature, user_input_humidity):
    # Create a dataframe with user input
    user_df = pd.DataFrame([[user_input_temperature, user_input_humidity]], columns=['Temperature', 'Humidity'])

    # Predict for the next hour using the loaded model
    next_hour_prediction = loaded_model.predict(user_df)
    return next_hour_prediction

# Streamlit UI components
st.title('Temperature and Humidity Prediction')
user_input_temperature = st.number_input("Enter the current temperature:")
user_input_humidity = st.number_input("Enter the current humidity:")

if st.button('Predict'):
    # Predict next hour values
    prediction = predict_next_hour(user_input_temperature, user_input_humidity)
    
    # Output the predicted values for the next hour
    st.write("Predicted temperature and humidity for the next hour:")
    st.write(f"Temperature: {prediction[0][0]}")
    st.write(f"Humidity: {prediction[0][1]}")
