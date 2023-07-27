from utils import util
import keras
stockDf = util.CreatePdDataframeForSingleStockPrice()
#util.PrintDF(stockDf,'Open')
trainData, testData, valData = util.CreateTrainTestValidationSet(stockDf)

util.PrintDF(trainData,"Open")
util.PrintDF(testData,"Open")
util.PrintDF(valData,"Open")

num_features = 5
num_categories = 1
"""
#TODO Create:
    X_train: Input data (Open, High, Low, Volume) <--- trainData
    y_train: output data (Close) <--- trainData
    
    X_test: Input data (Open, High, Low, Volume) <--- testData
    y_test: output data (Close) <--- testData
    
    X_val: Input data (Open, High, Low, Volume) <--- valData
    y_val: output data (Close) <--- valData
    
"""

x_train = trainData[['Open', 'High', 'Low', 'Volume']].values
y_train = trainData['Close'].values
x_test = testData[['Open', 'High', 'Low', 'Volume']].values
y_test = testData['Close'].values
x_val = valData[['Open', 'High', 'Low', 'Volume']].values
y_val = valData['Close'].values

# Create the model
model = util.CreateModel(num_features, num_categories)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=25, batch_size=50, validation_data=(x_val, y_val), verbose=1)

# Evaluate the model
loss = model.evaluate(x_test, y_test)
print("Test Loss: ", loss)



