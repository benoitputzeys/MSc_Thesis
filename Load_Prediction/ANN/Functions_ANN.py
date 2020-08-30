
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

    # Input shape corresponds to the number of columns (the features EMA, SMA, transmission) of the dataframe
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

    # Define the optimizer
    opt = keras.optimizers.Adam(lr=learning_rate)

    # Compile the model by defining the loss with respect to which will be optimized.
    # Also include metrics that can be used to see how they are reduced during training.
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model


def train_model(model, xvalues, yvalues, epochs, batch):

    # Actually train the model with the inputs x and target values y.
    history = model.fit(xvalues, yvalues, batch_size=batch, epochs=epochs, verbose = 2)
    # To track the progression of training, gather a snapshot of the model's mean absolute error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist
