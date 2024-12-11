import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf

# Load the stock data from Yahoo Finance
# yf.pdr_override()

start = "2009-01-01"
end = "2023-01-01"

st.title('Stock Closing Price Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'GOOGL')
df = yf.download(user_input, start, end)

st.subheader('Data from 1st Jan, 2009 to 1st Jan, 2023')
st.write(df.describe())

# Plot the closing price
st.subheader('Closing Price Vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig1)

# Plot moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Plot with 100 days moving average
st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'r', label="Per Day Closing")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig2)

# Plot with both 100 days and 200 days moving averages
st.subheader('Closing Price Vs Time Chart with 100 days and 200 days Moving Average')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(ma200, 'b', label="Moving Average 200")
plt.plot(ma100, 'g', label="Moving Average 100")
st.pyplot(fig3)

# Data preprocessing
train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):])

scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)

# Preparing data for model input
x_train = []
y_train = []
for i in range(100, train_df_arr.shape[0]):
    x_train.append(train_df_arr[i-100:i, 0])
    y_train.append(train_df_arr[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Prepare test data
past_100_days = train_df.tail(100)
final_df = past_100_days._append(test_df, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i, 0])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Model prediction
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Final plot with predicted and original prices
st.subheader('Predicted Vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
st.pyplot(fig4)
