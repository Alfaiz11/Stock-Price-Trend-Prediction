# Stock Price Trend Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock price trends based on historical data. It also integrates technical indicators like Moving Average (MA) and Relative Strength Index (RSI) for better trend analysis.

---

## Features

- LSTM-based price prediction
- Moving Average (MA20) overlay
- RSI indicator (20-day)
- Interactive Streamlit dashboard
- Input custom stock ticker and date range

---

## Tech Stack

- Python, Streamlit, yfinance, TensorFlow/Keras
- Pandas, NumPy, scikit-learn, matplotlib
- `ta` (technical analysis library)

---

## Run Locally

```bash
git clone https://github.com/yourusername/stock-trend-prediction-lstm.git
cd stock-trend-prediction-lstm
pip install -r requirements.txt
streamlit run app.py
