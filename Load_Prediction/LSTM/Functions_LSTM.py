import pandas as pd
import keras

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def create_model(input_variable, learning_rate):
    # Initialising the NN
    # Initialise the NN as a sequence of layers as opposed to a computational graph.
    my_model = Sequential()

    my_model.add(LSTM(units=25, return_sequences=True, input_shape=(input_variable.shape[1],1), kernel_initializer='uniform'))
    my_model.add(Dropout(0.25))
    my_model.add(LSTM(units=25,return_sequences=True,  kernel_initializer='uniform'))
    my_model.add(Dropout(0.25))
    my_model.add(LSTM(units = 10, kernel_initializer='uniform'))
    my_model.add(Dropout(0.25))
    my_model.add(Dense(units=1, kernel_initializer='uniform'))

    # Compiling the RNN
    opt = keras.optimizers.Adam(lr=learning_rate)
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model

def train_model(model, xvalues, yvalues, epochs, batch):

    # Fitting the model to the Training set
    history = model.fit(xvalues, yvalues, epochs=epochs, batch_size=batch, verbose = 2)

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist
