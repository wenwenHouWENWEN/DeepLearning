import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Disable GPU, use CPU only
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Print TensorFlow and Keras version
print("TensorFlow version:", tf.__version__)
print("Keras version from TensorFlow:", tf.keras.__version__)
print("Available devices:", tf.config.list_physical_devices())

# Load and preprocess the data
file_path = '中欧医疗健康混合C_net_value_2years.csv'
fund_data = pd.read_csv(file_path)
fund_data['Date'] = pd.to_datetime(fund_data['Date'])
fund_data.sort_values(by='Date', inplace=True)
fund_data.fillna(method='ffill', inplace=True)

# Display a sample of the data
print("Data Sample:")
print(fund_data.head())

# Only take the net value for prediction
net_values = fund_data['Net Value'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_net_values = scaler.fit_transform(net_values)
plt.switch_backend('macosx')
# Split into training and testing sets
train_size = int(len(scaled_net_values) * 0.8)
train, test = scaled_net_values[:train_size], scaled_net_values[train_size:]


# Create dataset function
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 30  # Set sliding window size
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

# Reshape input data to [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=1, verbose=2)


# Predict the next 30 days
def predict_next_days(model, last_window, days, scaler):
    predictions = []
    current_window = last_window.copy()

    for _ in range(days):
        prediction = model.predict(current_window.reshape(1, look_back, 1))[0, 0]
        predictions.append(prediction)
        # Update the window with the new prediction
        current_window = np.roll(current_window, -1)
        current_window[-1, 0] = prediction

    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()


# Get the last window from the test set
last_window = scaled_net_values[-look_back:]
# Predict the next 30 days
future_days = 30
future_predictions = predict_next_days(model, last_window, future_days, scaler)

# Generate future dates
last_date = fund_data['Date'].iloc[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1, freq='D')[1:]

# Plot the future 30 days predictions
plt.figure(figsize=(12, 6))

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions', color='purple', linestyle='--', marker='o')

plt.xlabel('Date')
plt.ylabel('Net Value')
plt.title('Future 30 Days Net Value Prediction using LSTM')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Net Value': future_predictions})
print("Future Predictions for the next 30 days:")
print(future_df)