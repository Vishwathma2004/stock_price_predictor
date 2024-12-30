import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Set up the Streamlit app
st.title("Stock Price Predictor App")
stock = st.text_input("Enter the Stock Id(for NSE add .NS & for BSE add .BO)", "NATIONALUM.NS")

# Download stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
nalco_data = yf.download(stock, start, end)

st.subheader("Stock Data")
st.write(nalco_data)

# Calculate moving averages
nalco_data['MA_for_250_days'] = nalco_data['Close'].rolling(250).mean()
nalco_data['MA_for_200_days'] = nalco_data['Close'].rolling(200).mean()
nalco_data['MA_for_100_days'] = nalco_data['Close'].rolling(100).mean()

# Calculate RSI
nalco_data['RSI'] = calculate_rsi(nalco_data)

# Plot moving averages and RSI
def plot_graph(figsize, values, full_data):
    fig = plt.figure(figsize=figsize)
    plt.plot(full_data['Close'], label='Original Close Price', color='blue')
    plt.plot(values, label='Moving Average', color='orange')
    plt.legend()
    return fig

st.subheader('Original Close Price and MA for 250 days')
st.pyplot(plot_graph((15, 6), nalco_data['MA_for_250_days'], nalco_data))

st.subheader('Original Close Price and MA for 200 days')
st.pyplot(plot_graph((15, 6), nalco_data['MA_for_200_days'], nalco_data))

st.subheader('Original Close Price and MA for 100 days')
st.pyplot(plot_graph((15, 6), nalco_data['MA_for_100_days'], nalco_data))

# Prepare data for prediction with additional features
nalco_data['Prev_Close'] = nalco_data['Close'].shift(1)
nalco_data['MA_5'] = nalco_data['Close'].rolling(window=5).mean()
nalco_data['MA_10'] = nalco_data['Close'].rolling(window=10).mean()
nalco_data.dropna(inplace=True)  

# Include RSI in features
features = nalco_data[['Prev_Close', 'MA_5', 'MA_10', 'RSI']]
target = nalco_data['Close']

splitting_len = int(len(nalco_data) * 0.7)
x_train = features[:splitting_len]
y_train = target[:splitting_len]
x_test = features[splitting_len:]
y_test = target[splitting_len:]

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
predictions = model.predict(x_test)

# Create DataFrame for plotting predictions
plotting_data = pd.DataFrame({
    'original_test_data': y_test.values.flatten(),  
    'predictions': predictions.flatten()  
}, index=y_test.index)

st.subheader('Original Values vs Predicted Values')
st.write(plotting_data)

# Calculate metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

# Plot predictions vs actual values
fig = plt.figure(figsize=(15, 6))
plt.plot(plotting_data.index, plotting_data['original_test_data'], label='Actual Prices', color='blue')
plt.plot(plotting_data.index, plotting_data['predictions'], label='Predicted Prices', color='orange')
plt.legend()
st.pyplot(fig)

# Prepare data for tomorrow's prediction
last_row = nalco_data.iloc[-1]
tomorrow_features = pd.DataFrame({
    'Prev_Close': [last_row['Close']],
    'MA_5': [nalco_data['MA_5'].iloc[-1]],  
    'MA_10': [nalco_data['MA_10'].iloc[-1]],  
    'RSI': [last_row['RSI']]  # Add RSI to features
})

tomorrow_prediction = model.predict(tomorrow_features)

tomorrow_date = datetime.now() + timedelta(days=1)
st.write(f'Predicted Stock Price for {tomorrow_date.strftime("%d-%m-%Y")}: {tomorrow_prediction[0][0]:.2f}')

# Prepare data for the day after tomorrow's prediction
day_after_tomorrow_features = pd.DataFrame({
    'Prev_Close': [tomorrow_prediction[0]],  
    'MA_5': [nalco_data['MA_5'].iloc[-1]],  
    'MA_10': [nalco_data['MA_10'].iloc[-1]],  
    'RSI': [tomorrow_features['RSI'][0]]  # Use tomorrow's predicted RSI if needed or last known RSI
})

day_after_tomorrow_prediction = model.predict(day_after_tomorrow_features)

day_after_tomorrow_date = datetime.now() + timedelta(days=2)
st.write(f'Predicted Stock Price for {day_after_tomorrow_date.strftime("%d-%m-%Y")}: {day_after_tomorrow_prediction[0][0]:.2f}')

# Prepare data for three days ahead prediction
three_days_ahead_features = pd.DataFrame({
    'Prev_Close': [day_after_tomorrow_prediction[0]],  
    'MA_5': [nalco_data['MA_5'].iloc[-1]],  
    'MA_10': [nalco_data['MA_10'].iloc[-1]],  
    'RSI': [day_after_tomorrow_features['RSI'][0]] 
})

three_days_ahead_prediction = model.predict(three_days_ahead_features)

three_days_ahead_date = datetime.now() + timedelta(days=3)
st.write(f'Predicted Stock Price for {three_days_ahead_date.strftime("%d-%m-%Y")}: {three_days_ahead_prediction[0][0]:.2f}')

# Prepare data for four days ahead prediction
four_days_ahead_features = pd.DataFrame({
    'Prev_Close': [three_days_ahead_prediction[0]],  
    'MA_5': [nalco_data['MA_5'].iloc[-1]],  
    'MA_10': [nalco_data['MA_10'].iloc[-1]],  
    'RSI': [three_days_ahead_features['RSI'][0]] 
})

four_days_ahead_prediction = model.predict(four_days_ahead_features)

four_days_ahead_date = datetime.now() + timedelta(days=4)
st.write(f'Predicted Stock Price for {four_days_ahead_date.strftime("%d-%m-%Y")}: {four_days_ahead_prediction[0][0]:.2f}')

# Prepare data for five days ahead prediction
five_days_ahead_features = pd.DataFrame({
    'Prev_Close': [four_days_ahead_prediction[0]],  
    'MA_5': [nalco_data['MA_5'].iloc[-1]],  
    'MA_10': [nalco_data['MA_10'].iloc[-1]],  
    'RSI': [four_days_ahead_features['RSI'][0]] 
})

five_days_ahead_prediction = model.predict(five_days_ahead_features)

five_days_ahead_date = datetime.now() + timedelta(days=5)
st.write(f'Predicted Stock Price for {five_days_ahead_date.strftime("%d-%m-%Y")}: {five_days_ahead_prediction[0][0]:.2f}')
