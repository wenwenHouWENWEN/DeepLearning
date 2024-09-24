import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set matplotlib to use macOS backend
plt.switch_backend('macosx')

# Disable all GPUs and use only CPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

# Print TensorFlow and Keras version information
print("TensorFlow version:", tf.__version__)
print("Keras version from TensorFlow:", tf.keras.__version__)
print("Available devices:", tf.config.list_physical_devices())

# Read and preprocess data
file_path = '中欧医疗健康混合C_net_value_2years.csv'
fund_data = pd.read_csv(file_path)
fund_data['Date'] = pd.to_datetime(fund_data['Date'])
fund_data.sort_values(by='Date', inplace=True)
fund_data.fillna(method='ffill', inplace=True)

# Print data sample
print("Data Sample:")
print(fund_data.head())

# Set sliding window and train/test data
window_size = 30
split_index = int(len(fund_data) * 0.8)
train_data = fund_data[:split_index]
test_data = fund_data[split_index:]

print(f"Training data length: {len(train_data)}")
print(f"Testing data length: {len(test_data)}")

# Create dataset using sliding window
def create_dataset(data, window_size):
    X = []
    Y = []
    for i in range(len(data) - window_size):
        X.append(data['Net Value'].iloc[i:i + window_size].values)
        Y.append(data['Net Value'].iloc[i + window_size])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Create training and test datasets
X_train, Y_train = create_dataset(train_data, window_size)
X_test, Y_test = create_dataset(test_data, window_size)

print(f"Training set length after windowing: {len(X_train)}")
print(f"Testing set length after windowing: {len(X_test)}")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Build a simple neural network model
def build_simple_model(input_dim):
    inputs = Input(shape=(input_dim,))
    hidden = Dense(64, activation='relu')(inputs)
    hidden = Dense(32, activation='relu')(hidden)
    outputs = Dense(1)(hidden)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

# Build and train the model
model = build_simple_model(input_dim=X_train.shape[1])
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

# Predict on the test set
predictions = model.predict(X_test)

# Compute prediction errors and standard deviation
prediction_errors = Y_test - predictions.flatten()
std_error = np.std(prediction_errors)

# Plot histogram of prediction errors
plt.figure(figsize=(12, 6))
plt.hist(prediction_errors, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.axvline(x=std_error, color='blue', linestyle='--', label=f'Standard Error = {std_error:.5f}')
plt.axvline(x=-std_error, color='blue', linestyle='--')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.legend()
plt.grid(True)
plt.show()

# Plot predicted vs actual values
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, predictions.flatten(), color='blue', alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Net Value')
plt.ylabel('Predicted Net Value')
plt.title('Predicted vs Actual Net Value')
plt.legend()
plt.grid(True)
plt.show()

# Predict future values and calculate confidence intervals
def predict_future_values_with_ci(model, last_window, days, std_error, z_score=1.96):
    future_predictions = []
    lower_bounds = []
    upper_bounds = []
    current_window = last_window.copy()

    for _ in range(days):
        prediction = model.predict(current_window.reshape(1, -1))[0, 0]
        future_predictions.append(prediction)

        # Calculate confidence intervals
        lower_bounds.append(prediction - z_score * std_error)
        upper_bounds.append(prediction + z_score * std_error)

        # Update the window for the next prediction
        current_window = np.roll(current_window, -1)
        current_window[-1] = prediction

    return np.array(future_predictions), np.array(lower_bounds), np.array(upper_bounds)

# Use the last window to predict future values
last_window = test_data['Net Value'].values[-window_size:]
future_days = 30  # Predict for the next 30 days
future_predictions, lower_bounds, upper_bounds = predict_future_values_with_ci(model, last_window, future_days, std_error)

# Generate future dates
last_date = fund_data['Date'].iloc[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1, freq='D')[1:]

# Plot future predictions with confidence intervals
plt.figure(figsize=(14, 7))

# Future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions', color='blue', linestyle='--')

# Confidence intervals
plt.fill_between(future_dates, lower_bounds, upper_bounds, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Date')
plt.ylabel('Net Value')
plt.title('Future Net Value Predictions for 1 Month with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.show()

# Print future predictions with confidence intervals
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Net Value': future_predictions,
    'Lower Bound': lower_bounds,
    'Upper Bound': upper_bounds
})
print("Future Predictions with Confidence Intervals for the next 30 days:")
print(future_df)

# Save results to CSV file
future_df.to_csv('future_df_predictions_with_ci.csv', index=False)