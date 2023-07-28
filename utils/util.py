#Thanks to the contributions made by...
#https://huggingface.co/blog/time-series-transformers
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# DP - original dates 2010-1-1 to 2023-7-1, scale down for model
# model like <7 years of data
def CreatePdDataframeForSingleStockPrice(tickerSymbol = 'TSLA', startDate='2020-1-1', endDate='2023-7-1', period = '1d'):
    tickerSymbol = tickerSymbol
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(period=period, start=startDate, end=endDate)
    return tickerDf
    
def PrintDF(df,dfTitle):
    df[dfTitle].plot(kind='line')
    plt.show()
    
    
def CreateTrainTestValidationSet(df, trainPercent = 0.7, validationPercent=0.2):
    train_length = int(len(df) * trainPercent)
    valid_length = int(len(df) * validationPercent)
   
    train_dataset = df[:train_length]
    valid_dataset = df[train_length:train_length + valid_length]
    test_dataset = df[train_length + valid_length:]
    
    return train_dataset, valid_dataset, test_dataset

    
def CreateTorchDataLoader(trainData, testData, validationData, batchSize = '32'):
      train_loader = DataLoader(trainData, batch_size=batchSize, shuffle=False)
      test_loader = DataLoader(testData, batch_size=batchSize, shuffle=False)  
      valid_loader = DataLoader(validationData, batch_size=batchSize, shuffle=False)
      
      return train_loader, test_loader, valid_loader
  
def CreateModel(NumOfFeatures, NumOfCategories):
    num_output_categories = NumOfCategories
    #initializer = tf.keras.initializers.GlorotUniform(seed)
    initializer = tf.keras.initializers.GlorotUniform()

    model = Sequential([
        #Flatten(input_shape=(256, 1)),
        # dense layer 1
        Dense(NumOfFeatures, activation='selu',kernel_initializer=initializer),  
        # dense layer 2
        Dense(512, activation='selu'), 
        # dense layer 3
        Dense(1024, activation='selu'), 
        # dense layer 3
        Dense(512, activation='selu'), 
        # dense layer 3
        Dense(256, activation='selu'),
        # dense layer 4
        Dense(64, activation='selu'), 
        # dense layer 5
        Dense(32, activation='selu'), 
        # dense layer 6
        Dense(16, activation='selu'),
        # output layer
        Dense(num_output_categories, activation='softmax'),  
    ])
    
    return model
