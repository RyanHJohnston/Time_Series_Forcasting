from utils import util
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

stockDf = util.CreatePdDataframeForSingleStockPrice()
trainData, testData, valData = util.CreateTrainTestValidationSet(stockDf)

#Training Data
train_data = trainData.iloc[:,1:2].values

scaler = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(train_data)

X_train = []
y_train = []
timeStep = 60
for i in range(timeStep, len(train_data)):
    X_train.append(training_set_scaled[i-timeStep:i,0])
    y_train.append(training_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
#Validation Data
val_data = valData.iloc[:,1:2].values
val_set_scaled = scaler.transform(val_data)

X_val = []
y_val = []
for i in range(timeStep, len(val_data)):
    X_val.append(val_set_scaled[i-timeStep:i,0])
    y_val.append(val_set_scaled[i,0])

X_val, y_val = np.array(X_val), np.array(y_val)
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

#Building the LSTM Model
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#Compiling and fitting the model
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x=X_train, y=y_train, batch_size=10, epochs=100, validation_data=(X_val, y_val))

#Testing the Model
test_data = testData.iloc[:,1:2].values
test_set_scaled = scaler.transform(test_data)

X_test = []
y_test = []
for i in range(timeStep, len(test_data)):
    X_test.append(test_set_scaled[i-timeStep:i,0])
    y_test.append(test_set_scaled[i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
print('Root Mean Squared Error:', rmse)

plt.figure(figsize=(14, 5))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price , color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
