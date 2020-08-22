
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.layers import Dense
from keras.layers import Dropout
import datetime
from pandas import DataFrame


def create_model(dim, learning_rate):

    # Create the model.
    my_model = keras.Sequential()

    # Input shape corresponds to the number of columns (the features day, month and year) of the dataframe,
    # excpet for the output label, the temperature.
    my_model.add(Dense(units=300, kernel_initializer='uniform', input_dim = dim, activation='relu'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=275, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=250, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=250, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=75, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

    opt = keras.optimizers.Adam(lr=learning_rate)
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model


def train_model(model, xvalues, yvalues, epochs, batch):

    history = model.fit(xvalues, yvalues, batch_size=batch, epochs=epochs, verbose = 2)

    # To track the progression of training, gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist
