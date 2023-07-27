import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.autograd import Variable
from torch.nn import Linear
from torch.nn import ReLU, CrossEntropyLoss, Sequential
from torch.nn import Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d
from torch.nn import Dropout
from torch.optim import Adam, SGD

#not tested for errors
#hardware issues

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(

            Conv2d(1, 4, kernel_size=3, stride=1, padding=1 ),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
            )

        self.linear_layers = Sequential(
            Linear(2359296, 1, 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view( -1, 2359296)
        x = self.linear_layers(x)

        return x


class Neural_Network:
    def __init__(self):
        pass

    def get_all_images(self, path):
        os.chdir(path)
        files = os.listdir(path)

        images = []

        counter = 0

        for file in files:
            file_path = path + '/' + file
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (1024, 1024))
            image = image.astype('float32')
            images.append(image)
            counter += 1

            if counter > 4:
                break

        return images


    def train_test_split(self, arr, train_percentage, label):
        train_size = int(train_percentage * len(arr))
        test_size = len(arr) - train_size

        train_x = arr[:train_size]
        test_x = arr[train_size:]

        train_y = []
        test_y = []

        for i in range(0, train_size):
            train_y.append(label)
        for i in range(train_size, len(arr)):
            test_y.append(label)

        train_x     = np.asarray(train_x)
        test_x      = np.asarray(test_x)
        train_y     = np.asarray(train_y)
        test_y      = np.asarray(test_y)

        return train_x, test_x,  train_y, test_y



    def handle_data(self, X_train, X_test, Y_train, Y_test):

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[1])
        X_train = torch.from_numpy(X_train)

        Y_train = Y_train.astype(int)
        Y_train = torch.from_numpy(Y_train)

        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[1])
        X_test = torch.from_numpy(X_test)

        Y_test = Y_test.astype(int)
        Y_test = torch.from_numpy(Y_test)

        return X_train, X_test, Y_train, Y_test


    def CNN(self, Normal, Covid, Vp):

        normal_X_train, normal_X_test, normal_Y_train, normal_Y_test = self.train_test_split(Normal, .7, 0)
        covid_X_train, covid_X_test, covid_Y_train, covid_Y_test = self.train_test_split(Covid, .7, 1)
        vp_X_train, vp_X_test, vp_Y_train, vp_Y_test = self.train_test_split(Vp, .7, 2)

        X_train = np.concatenate((normal_X_train, covid_X_train, vp_X_train))
        X_test = np.concatenate((normal_X_test, covid_X_test, vp_X_test))
        Y_train = np.concatenate((normal_Y_train, covid_Y_train, vp_Y_train))
        Y_test = np.concatenate((normal_Y_test, covid_Y_test, vp_Y_test))

        X_train, X_test, Y_train, Y_test = self.handle_data(X_train, X_test, Y_train, Y_test)

        model = Net()
        optimizer = Adam(model.parameters(), lr = 0.07)
        criterion = CrossEntropyLoss()

        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()


        n_epochs = 25


        for epoch in range(1, n_epochs) :
            self.train(epoch, model, X_train, X_test, Y_train, Y_test, optimizer, criterion)


    def train(self, epoch, model, trainx, testx, trainy, testy, optimizer, criterion):
        model.train()

        tr_loss = 0
        train_losses = []
        test_losses = []

        X_train, Y_train = Variable(trainx), Variable(trainy)
        X_test, Y_test = Variable(testx), Variable(testy)

        if torch.cuda.is_available():
            X_train = X_train.cuda()
            X_test = X_test.cuda()
            Y_train = Y_train.cuda()
            Y_test = Y_test.cuda()

        optimizer.zero_grad()

        output_train = model(X_train)
        output_test = model(X_test)

        loss_train = criterion(output_train, Y_train)
        loss_test = criterion(output_test, Y_test)

        train_losses.append(loss_train)
        test_losses.append(loss_test)

        loss_train.backward()
        optimizer.step()

        tr_loss = loss_train.item()


        if epoch % 2 == 0:

            print("Epoch = ", epoch + 1, "\tLoss = ", loss_test)


        plt.plot(train_losses, label = 'Training loss')
        plt.plot(val_losses, label = 'Validation loss')
        plt.legend()
        plt.show()


        #training set predictions accuracy
        with torch.no_grad():
            output = model(X_train.cuda())

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis = 1)

        accuracy_score(Y_train, predictions)


        #test set prediction accuracy

        with torch.no_grad():
            output = model(X_test.cuda())

        softmax = torch.exp(output).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis = 1)

        accuracy_score(Y_test, predictions)

        




def main():

    home = '/users/satish/Desktop/COVID'

    os.chdir(home)

    model = Neural_Network()

    Normal = model.get_all_images(home + '/NORMAL' )
    Covid = model.get_all_images(home + '/COVID' )
    Vp = model.get_all_images(home  + '/Viral Pneumonia')



    model.CNN(Normal, Covid, Vp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()





