import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Disable GPU, use CPU only
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')

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
plt.switch_backend('macosx')
# Display a sample of the data
print("Data Sample:")
print(fund_data.head())

# Set sliding window and create training/testing datasets
window_size = 30
split_index = int(len(fund_data) * 0.8)
train_data = fund_data[:split_index]
test_data = fund_data[split_index:]

print(f"Training data length: {len(train_data)}")
print(f"Testing data length: {len(test_data)}")

# Function to create datasets with sliding window
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data['Net Value'].iloc[i:i + window_size].values)
        Y.append(data['Net Value'].iloc[i + window_size])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Create training and testing datasets
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

# Predict on training and testing datasets
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate RMSE for training and testing datasets
train_rmse = np.sqrt(np.mean((train_predictions.flatten() - Y_train) ** 2))
print(f"RMSE on the training set: {train_rmse:.4f}")

test_rmse = np.sqrt(np.mean((test_predictions.flatten() - Y_test) ** 2))
print(f"RMSE on the test set: {test_rmse:.4f}")

# Plot actual values and predictions
plt.figure(figsize=(14, 7))

# Original data
plt.plot(fund_data['Date'], fund_data['Net Value'], label='Original Net Value', color='blue')

# Train predictions
train_predict_plot = np.empty_like(fund_data['Net Value'])
train_predict_plot[:] = np.nan
train_predict_plot[window_size:len(train_predictions) + window_size] = train_predictions.flatten()

# Correct the start and end indices for test predictions
start_test_idx = len(train_data)
end_test_idx = start_test_idx + len(test_predictions)

test_predict_plot = np.empty_like(fund_data['Net Value'])
test_predict_plot[:] = np.nan
test_predict_plot[start_test_idx:end_test_idx] = test_predictions.flatten()

# Plot predictions
plt.plot(fund_data['Date'], train_predict_plot, label='Train Predict', color='red')
plt.plot(fund_data['Date'], test_predict_plot, label='Test Predict', color='green')

plt.xlabel('Date')
plt.ylabel('Net Value')
plt.title('Net Value Prediction using Simple Neural Network')
plt.legend()
plt.grid(True)
plt.show()