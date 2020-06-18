import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop

def create_model(input_variable, learning_rate):
    # Initialising the NN
    # Initialise the NN as a sequence of layers as opposed to a computational graph.
    my_model = Sequential()
    initializer = RandomNormal(mean = 0., stddev = 1.)

    # Because predicting the temperature is pretty complex, you need to have a high dimensionality too thus 50
    # for the number of neurons. If the number of neurons is too small in each of the LSTM layers, the model would not
    # capture very well the upward and downward trend.
    my_model.add(LSTM(units=50, return_sequences=True, input_shape=(input_variable.shape[1],1),
                       kernel_initializer=initializer))
    my_model.add(Dropout(0.1))

    # Adding a second LSTM layer and Dropout regularisation
    # No need to specify any input shape here because we have already defined that we have 50 neurons in the
    # previous layer.
    my_model.add(LSTM(units=50,return_sequences=True,  kernel_initializer=initializer))
    my_model.add(Dropout(0.1))

    # Adding a third LSTM layer and Dropout regularisation
    my_model.add(LSTM(units=50,return_sequences=True,  kernel_initializer=initializer))
    my_model.add(Dropout(0.1))

    # Adding a fourth LSTM layer and Dropout regularisation
    # This is the last LSTM layer that is  added! Thus the return sequences is set to  false.
    my_model.add(LSTM(units = 25, kernel_initializer=initializer))
    my_model.add(Dropout(0.1))

    # Adding the output layer
    # We are not adding an LSTM layer. We are fully connecting the outward layer to the previous LSTM layer.
    # As a result, we use a DENSE layer to make this full connection.
    my_model.add(Dense(units=1, kernel_initializer=initializer))

    # Compiling the RNN
    # For RNN and also in the Keras documentation, an RMSprop is recommended.
    # But experimenting with other optimizers, one can also use the adam optimizer.
    # The adam optimizer is actually always a good choice and very powerfull too!
    # In general, the most commonly used optimizers are adam and RMSprop
    optimizer = RMSprop(lr=learning_rate)
    my_model.compile(optimizer=optimizer, loss='mean_squared_error')

    return my_model

def train_model(model, xvalues, yvalues, epochs, batch):

    # Fitting the model to the Training set
    history = model.fit(xvalues, yvalues, epochs=epochs, batch_size=batch)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    loss = hist["loss"]

    return epochs, loss

def plot_loss(epochs, difference):


    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.plot(epochs, difference)
    plt.legend()
    plt.show()

    return None

def plot_generation(ax, y_values, string):
    ax.plot(y_values, linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(string)
    plt.show()

def plot_prediction_zoomed_in( yvalues1, yvalues2, yvalues3, string1, string2, string3 ):
    plt.figure(4)
    plt.xlabel("Time")
    plt.ylabel("Generation")
    plt.plot( yvalues1 , label=string1)
    plt.plot( yvalues2 , label=string2)
    plt.plot( yvalues3 , label=string3)
    plt.legend()
    plt.show()
