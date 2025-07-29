import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import ta

# User Input

st.sidebar.title("Stock Price Trend Prediction")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))


# Fetch Data

st.title("Stock Trend Forecast with LSTM")

data = yf.download(ticker, start=start_date, end=end_date)
if data.empty:
    st.error("Failed to fetch data. Try another ticker.")
    st.stop()

st.subheader(f"Raw Data for {ticker}")
st.line_chart(data['Close'])

# Feature Engineering
data['Close'] = data['Close'].astype(float)

# Moving Average
data['MA20'] = data['Close'].rolling(window=20).mean()

# Ensure RSI input is 1D Series
rsi_input = pd.Series(data['Close'].values.flatten(), index=data.index)
rsi_indicator = ta.momentum.RSIIndicator(close=rsi_input, window=14)
data['RSI'] = rsi_indicator.rsi()

# Drop rows with any NaNs
data.dropna(inplace=True)



# Features: Close, MA20, RSI
features = data[['Close', 'MA20', 'RSI']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# Prepare sequences
time_step = 60
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i])
    y.append(scaled_data[i][0])
X, y = np.array(X), np.array(y)

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)

# Predict and Reverse Scale
predicted_scaled = model.predict(X_test)
predicted = scaler.inverse_transform(np.hstack([predicted_scaled, np.zeros((len(predicted_scaled), 2))]))[:, 0]
actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))]))[:, 0]

# Plot Results
st.subheader("Predicted vs Actual Close Price")
fig1 = plt.figure(figsize=(12, 5))
plt.plot(actual, label='Actual Price', color='blue')
plt.plot(predicted, label='Predicted Price', color='orange')
plt.legend()
plt.title("Actual vs Predicted Close Price")
st.pyplot(fig1)

# MA and Close
st.subheader("Close Price with 20-Day Moving Average")
fig2 = plt.figure(figsize=(12, 4))
plt.plot(data['Close'], label='Close', color='blue')
plt.plot(data['MA20'], label='MA20', color='green')
plt.legend()
st.pyplot(fig2)

# RSI Chart
st.subheader("RSI Indicator")
fig3 = plt.figure(figsize=(12, 3))
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title("Relative Strength Index")
plt.legend()
st.pyplot(fig3)

# Footer
st.markdown("Predictions are for trend analysis only.")
