import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load the stock data into a Pandas DataFrame
df = pd.read_csv('stock_data.csv')

# Set the date column as the index
df.set_index('Date', inplace=True)

# Define the number of days to use for the prediction
prediction_days = 30

# Scale the data using the MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Split the data into training and testing sets
training_data = scaled_data[:-prediction_days]
test_data = scaled_data[-prediction_days:]

# Define the function to create the input and output sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Create the input and output sequences for the training data
X_train, y_train = create_sequences(training_data, prediction_days)

# Create the neural network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Create the input and output sequences for the test data
X_test, y_test = create_sequences(test_data, prediction_days)

# Make predictions on the test data
predictions = model.predict(X_test)

# Inverse scale the predictions and the actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# Plot the predicted and actual values
plt.plot(actual_values, color='red', label='Actual Stock Price')
plt.plot(predictions, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
