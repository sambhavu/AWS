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

"""
Solar Cycle Datasets SWPC 
"""

f10_flux_smoothed_json = 'https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux-smoothed.json'
f10_flux_json = 'https://services.swpc.noaa.gov/json/solar-cycle/f10-7cm-flux.json'
observed_solar_cycle_json = 'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json'
predicted_solar_cycle_json = 'https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json'
f10_flux_predicted_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-f10-7cm-flux-predicted-high-low.json'
predicted_solar_cycle_25_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-predicted.json'
predicted_solar_cycle_25_high_low_json = 'https://services.swpc.noaa.gov/json/solar-cycle/solar-cycle-25-ssn-predicted-high-low.json'
sunspots_smoothed_json = 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots-smoothed.json'
sunspots_json = 'https://services.swpc.noaa.gov/json/solar-cycle/sunspots.json'
swpc_observed_ssn_json = 'https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json'



class Solar():
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

    def print_double_arrays(self, arr1, arr2, arr1_name = "", arr2_name = ""):
          print("Arr1 Length => ", len(arr1))
          print("Arr2 Length => ", len(arr2))
          for i in range(0, len(arr1)):
              print(i, " ::",arr1_name, " ==> ", arr1[i], "::", arr2_name ," ==> ", arr2[i])


    def get_json_data(self, json_url):
        data = json.loads(urlopen(json_url).read())
        return data

    def get_json_array(self, data, tag):
        array = []

        try:
            for i in range(0, len(data)):
                array.append(data[i][tag])

        except Exception as e:
            print("Failed to create Dataset from JSON Data")
            print(e)

        return array

    def scale_data(self, array):
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

    def predict(self):
        predictions = self.model.predict(self.test_X)
        return predictions

    def handle_results(self, scaled_predictions):
        predictions = self.scaler.inverse_transform(scaled_predictions)
        actuals = self.scaler.inverse_transform([self.test_Y])

        predictions = predictions.reshape(1, predictions.shape[0])
        predictions = np.ravel(predictions)

        actuals = actuals.reshape(1, actuals.shape[1])
        actuals = np.ravel(actuals)

        return predictions, actuals

    def run_simple_prediction(self, json_link, json_attribute, train_test_split_percentage):

        array = self.get_json_array(self.get_json_data(json_link), json_attribute)
        scaled = self.scale_data(array)

        self.train_test_split(scaled, train_test_split_percentage)

        self.fit(10,1,1)
        predictions = self.predict()

        predictions, actuals = self.handle_results(predictions)

        return predictions, actuals




def main():

    f10_attribute = "f10.7"
    train_test_split_percentage = .95


    model = Solar()




    predictions, actuals = model.run_simple_prediction(f10_flux_json, f10_attribute, train_test_split_percentage)
    #plotting & graphing

    model.print_double_arrays(predictions, actuals)
    print(mean_squared_error(actuals, predictions, squared=False))
    plt.plot(actuals, label="actual")
    plt.plot(predictions, label="predictions")
    plt.legend(loc="upper left")
    plt.show()








main()
