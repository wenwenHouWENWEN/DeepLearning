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

# Compute standard deviation of prediction errors
prediction_errors = Y_test - predictions.flatten()
std_error = np.std(prediction_errors)

# Plot actual and predicted values
plt.figure(figsize=(14, 7))

# Actual values
plt.plot(test_data['Date'].iloc[window_size:], Y_test, label='Actual Net Value', color='green')

# Predicted values
plt.plot(test_data['Date'].iloc[window_size:], predictions, label='Predicted Net Value', color='red')

plt.xlabel('Date')
plt.ylabel('Net Value')
plt.title('Simple NN Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Print sliding window processed data
print("Sliding Window Processed Data (Train):")
print(pd.DataFrame(X_train).head())
print("Sliding Window Processed Data (Test):")
print(pd.DataFrame(X_test).head())

# Predict future values using Monte Carlo simulation and calculate confidence intervals
def predict_future_values_with_mc(model, last_window, days, num_samples=1000, z_score=1.96):
    future_predictions = []
    lower_bounds = []
    upper_bounds = []
    current_window = last_window.copy()

    for day in range(days):
        sample_predictions = []
        for _ in range(num_samples):
            # Add noise to the current window for each sample
            noise = np.random.normal(0, std_error, size=current_window.shape)
            noisy_window = current_window + noise
            prediction = model.predict(noisy_window.reshape(1, -1), verbose=0)[0, 0]
            sample_predictions.append(prediction)

        # Calculate mean and confidence intervals for the sampled predictions
        mean_prediction = np.mean(sample_predictions)
        std_dev_prediction = np.std(sample_predictions)

        future_predictions.append(mean_prediction)
        lower_bounds.append(mean_prediction - z_score * std_dev_prediction)
        upper_bounds.append(mean_prediction + z_score * std_dev_prediction)

        # Update the window for the next prediction
        current_window = np.roll(current_window, -1)
        current_window[-1] = mean_prediction

    return np.array(future_predictions), np.array(lower_bounds), np.array(upper_bounds)

# Use the last window to predict future values
last_window = test_data['Net Value'].values[-window_size:]
future_days = 30  # Predict for the next 30 days
future_predictions, lower_bounds, upper_bounds = predict_future_values_with_mc(model, last_window, future_days)

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
plt.title('Future Net Value Predictions for 1 Month with Monte Carlo Confidence Intervals')
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
print("Future Predictions with Monte Carlo Confidence Intervals for the next 30 days:")
print(future_df)

# Save results to CSV file
future_df.to_csv('future_df_predictions_with_mc_ci.csv', index=False)

# Plot the distribution of the first future date's predictions
plt.figure(figsize=(14, 7))
plt.hist(future_predictions[:num_samples], bins=50, alpha=0.7, label='Prediction Distribution for 2024-06-13')
plt.axvline(x=future_predictions[0], color='blue', linestyle='--', label='Mean Prediction')
plt.axvline(x=lower_bounds[0], color='red', linestyle='--', label='Lower Bound')
plt.axvline(x=upper_bounds[0], color='green', linestyle='--', label='Upper Bound')

plt.xlabel('Net Value')
plt.ylabel('Frequency')
plt.title('Prediction Distribution for the first future date (2024-06-13)')
plt.legend()
plt.grid(True)
plt.show()