import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import json
import urllib
from urllib.request import urlopen
import yfinance as yf



class Stock_Prediction():
    def __init__(self):

        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(1, 1), return_sequences=True))
        self.model.add(LSTM(16))
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.train_X = []
        self.train_Y = []
        self.test_X = []
        self.test_Y = []



    def print_array(self, array, header = ""):
        print(header)
        for i in range(0, len(array)):
            print(array[i])

    def print_double_arrays(self, arr1, arr2):
          print("Arr1 Length => ", len(arr1))
          print("Arr2 Length => ", len(arr2))
          for i in range(0, len(arr1)):
              print(i, " ::Prediction ==> ", arr1[i], "::Actual ==> ", arr2[i])

    def get_pandas_data_array(self, file_path, column_name):
        Dataframe = pd.read_csv(file_path)
        array = Dataframe[column_name].values
        return array

    def get_json_data(self, json_url):
        data = json.loads(urlopen(json_url).read())
        return data

    def get_data_array(self, data, tag):
        array = []

        try:
            for i in range(0, len(data)):
                array.append(data[i][tag])

        except Exception as e:
            print("Failed to create Dataset from JSON Data")
            print(e)

        return array

    def handle_data(self, array):
        array = np.reshape(array, (len(array), 1))
        array_scaled = self.scaler.fit_transform(array)
        return array_scaled

    def reshape_3d(self, array):
        return np.reshape(array, (array.shape[0], 1, array.shape[1]))



    def getdataset(self, dataset, step):
        X, Y = [], []
        for i in range(len(dataset) - step - 1):
            a = dataset[i:(i + step), 0]
            X.append(a)
            Y.append(dataset[i + step, 0])



        return np.array(X), np.array(Y)

    def train_test_split(self, scaled_array, split_percentage):
        trainSize = int(split_percentage * len(scaled_array))
        testSize = len(scaled_array) - trainSize

        trainClose, testClose = scaled_array[0:trainSize, :], scaled_array[: len(scaled_array), :]

        self.train_X, self.train_Y = self.getdataset(trainClose, 1)
        self.test_X, self.test_Y = self.getdataset(testClose, 1)

        self.train_X = self.reshape_3d(self.train_X)
        self.test_X = self.reshape_3d(self.test_X)


    def fit(self, epochs, batch_size, verbose):
        self.model.fit(self.train_X, self.train_Y, epochs = epochs, batch_size =batch_size,   verbose = verbose)

    def test(self):
        predictions = self.model.predict(self.test_X)
        return predictions

    def handle_results(self, scaled_predictions):
        predictions = self.scaler.inverse_transform(scaled_predictions)
        actuals = self.scaler.inverse_transform([self.test_Y])

        predictions = predictions.reshape(1, predictions.shape[0])
        predictions = np.ravel(predictions)

        actuals = actuals.reshape(1, actuals.shape[1])
        actuals = np.ravel(actuals)


        print("Predictions Shape: ", predictions.shape)
        print("actuals shape: ", actuals.shape)

        return predictions, actuals

    def run(self, array):
        scaled = self.handle_data(array)
        self.train_test_split(scaled, .7)
        self.fit(5,1,1)
        predictions = self.test()

        predictions, actuals = self.handle_results(predictions)

        self.trade(predictions, actuals)

        self.print_double_arrays(predictions, actuals)



        print(mean_squared_error(actuals, predictions, squared = False))


    def trade(self, predictions, actuals):

        account = []
        totalprofit = 0

        buy_everday = []
        buy_everday_profit = 0

        for i in range(1,len(actuals)):
            tomorrows_prediction = predictions[i]
            position = 0
            profit  = 0

            b_profit = 0

            if tomorrows_prediction > actuals[i-1]:
                position = 1

            else:
                position = -1

            profit = (actuals[i] - actuals[i-1]) * position

            bprofit = (actuals[i] - actuals[i-1])

            totalprofit = profit + totalprofit
            buy_everday_profit = bprofit + buy_everday_profit


            account.append(totalprofit)
            buy_everday.append(buy_everday_profit)


        plt.plot(account, label = "Account Profit")
        plt.plot(buy_everday, label = "Buy Everyday(for comparison) ")
        plt.legend(loc = "upper left")
        plt.show()





def main():

    csv_filepath = '/users/satish/Desktop/BTCUSD_1min.csv'
    
    model = Stock_Prediction()

    model.run(model.get_pandas_data_array('/users/satish/Desktop/BTCUSD_1min.csv', 'Close'))







main()
