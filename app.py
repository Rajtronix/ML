import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('datas.csv')  # Replace with the path to your CSV file

# Assuming your dataset has columns 'Temperature', 'Humidity', and 'Timestamp'
# You might need to adjust these column names to match your dataset
temperature_series = data['Temperature']
humidity_series = data['Humidity']

# ARIMA model for Temperature and Humidity
arima_temperature = ARIMA(temperature_series, order=(5, 1, 0))  # Example ARIMA order
arima_humidity = ARIMA(humidity_series, order=(5, 1, 0))

# LSTM model for Temperature and Humidity
scaler = MinMaxScaler()
temperature_series_scaled = scaler.fit_transform(temperature_series.values.reshape(-1, 1))
humidity_series_scaled = scaler.fit_transform(humidity_series.values.reshape(-1, 1))

sequence_length = 60  # Adjust this window size according to your preference

def prepare_data(series):
    data = []
    for i in range(len(series) - sequence_length):
        data.append(series[i:i+sequence_length])
    return np.array(data)

temperature_data = prepare_data(temperature_series_scaled)
humidity_data = prepare_data(humidity_series_scaled)

temperature_model = Sequential()
temperature_model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
temperature_model.add(LSTM(units=50))
temperature_model.add(Dense(units=1))
temperature_model.compile(optimizer='adam', loss='mean_squared_error')

humidity_model = Sequential()
humidity_model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)))
humidity_model.add(LSTM(units=50))
humidity_model.add(Dense(units=1))
humidity_model.compile(optimizer='adam', loss='mean_squared_error')

temperature_model.fit(temperature_data, temperature_data, epochs=100, batch_size=32)
humidity_model.fit(humidity_data, humidity_data, epochs=100, batch_size=32)

# Streamlit UI
st.title('Temperature and Humidity Predictor')

input_temp = st.number_input('Enter Temperature:')
input_hum = st.number_input('Enter Humidity:')

if st.button('Predict'):
    arima_temp_forecast = arima_temperature.forecast(steps=60)  # ARIMA prediction
    arima_hum_forecast = arima_humidity.forecast(steps=60)

    lstm_temp_forecast = temperature_model.predict(scaler.transform([[input_temp]]).reshape(1, sequence_length, 1))
    lstm_hum_forecast = humidity_model.predict(scaler.transform([[input_hum]]).reshape(1, sequence_length, 1))

    st.write(f'ARIMA Temperature Forecast for Next 1 Hour: {arima_temp_forecast[-1]}')
    st.write(f'ARIMA Humidity Forecast for Next 1 Hour: {arima_hum_forecast[-1]}')

    st.write(f'LSTM Temperature Forecast for Next 1 Hour: {lstm_temp_forecast[0, 0]}')
    st.write(f'LSTM Humidity Forecast for Next 1 Hour: {lstm_hum_forecast[0, 0]}')
