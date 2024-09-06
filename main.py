import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import h5py

data = pd.read_csv('data/stocks/a.us.txt', sep=",") #For reading txt file instead of csv

#print(len(data))

#print(data.head())

# Splitting data into training and test subsets
training_subset = data.iloc[:4000]  # First 80 rows for training
test_subset = data.iloc[4000:]  # Remaining rows for testing

# Normalizing the data for better convergence
train_mean = training_subset['Close'].mean()
train_std = training_subset['Close'].std()


train_scaled = (training_subset['Close'] - train_mean) / train_std

# Creating time series sequences for LSTM (can use any window size, e.g., 5)
window_size = 60
X_train = []
y_train = []

for i in range(window_size, len(train_scaled)):
    X_train.append(train_scaled[i-window_size:i])
    y_train.append(train_scaled[i])

X_train, y_train = np.array(X_train), np.array(y_train) #convert to numpy array

# Reshaping for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Building an LSTM model (you can also try SimpleRNN or GRU)
model = Sequential()
model.add(LSTM(60, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Fitting the model
#history = model.fit(X_train, y_train, epochs=10, verbose=1)
#model.save_weights('saved_weights.h5')

model.load_weights('saved_weights.h5')

# Extend the forecast window beyond test data (e.g., predict 100 more steps)
forecast_horizon = 600  # 520 for test data + 100 additional steps for future

# Forecasting for the next 8 periods
def forecast_lstm(model, input_seq, n_forecast, actual_data=None):
    forecast = []
    actual_data = actual_data.values if isinstance(actual_data, pd.Series) else actual_data  # Ensure it's an array
    for i in range(n_forecast):
        pred = model.predict(input_seq)
        forecast.append(pred[0, 0])
        # Use actual test data if provided, otherwise continue with the model's predictions
        if actual_data is not None and i < len(actual_data):
            input_seq = np.append(input_seq[0][1:], actual_data[i]).reshape(1, window_size, 1)
        else:
            input_seq = np.append(input_seq[0][1:], pred).reshape(1, window_size, 1)
    return np.array(forecast)

# Prepare the input sequence for forecasting
input_seq = np.array(train_scaled[-window_size:]).reshape(1, window_size, 1)

# Actual test scaled for comparison with the first part of the forecast
actual_test_scaled = (test_subset['Close'][:521] - train_mean) / train_std

# Forecast for a larger horizon
forecasted_values = forecast_lstm(model, input_seq, forecast_horizon, actual_test_scaled)

# Transform back to original scale (for plotting purposes)
forecasted_values = forecasted_values * train_std + train_mean

# Creating x-values (steps) for the extended forecast horizon
extended_steps = list(range(len(training_subset))) + list(range(len(training_subset), len(training_subset) + forecast_horizon))


# Plot the forecast against actual data, extending beyond test data
plt.figure(figsize=(10, 6))
plt.plot(training_subset.index, training_subset['Close'], label='Training Data')
plt.plot(test_subset.index[:521], test_subset['Close'], label='Test Data', color='green')
plt.plot(extended_steps[-forecast_horizon:], forecasted_values, label='LSTM Forecast', color='red')  # Extended forecast
plt.title('Forecast vs Actual Close Price (Extended Horizon)')
plt.xlabel('Step')
plt.ylabel('Close')
plt.legend()
plt.show()
