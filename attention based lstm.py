import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
import plotly.graph_objects as go

# Load and prepare data
df = pd.read_csv('/content/yahoo_stock.csv')  # Replace with your actual CSV
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create dataset function
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time steps and split data
time_step = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape to [samples, time steps, features]
X_train = X_train.reshape(-1, time_step, 1)
X_test = X_test.reshape(-1, time_step, 1)

# Build Bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, batch_size=1, epochs=10)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform([y_train])
y_test_actual = scaler.inverse_transform([y_test])

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train_actual[0], train_predict[:, 0]))
train_mae = mean_absolute_error(y_train_actual[0], train_predict[:, 0])
test_rmse = np.sqrt(mean_squared_error(y_test_actual[0], test_predict[:, 0]))
test_mae = mean_absolute_error(y_test_actual[0], test_predict[:, 0])

print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

# Plotting results
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict)+time_step, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(time_step*2)+1:len(scaled_data)-1, :] = test_predict

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df.index, y=train_plot[:, 0], name='Train Predict', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df.index, y=test_plot[:, 0], name='Test Predict', line=dict(color='red')))
fig.update_layout(title='Bidirectional LSTM Stock Price Prediction',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  template='plotly_dark')
fig.show()
