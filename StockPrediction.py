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
model = util.CreateModel(num_features, num_categories)




