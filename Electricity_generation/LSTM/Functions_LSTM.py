import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
import datetime
from pandas import DataFrame

def create_model(input_variable, learning_rate):
    # Initialising the NN
    # Initialise the NN as a sequence of layers as opposed to a computational graph.
    my_model = Sequential()

    # Because predicting the temperature is pretty complex, you need to have a high dimensionality too thus 50
    # for the number of neurons. If the number of neurons is too small in each of the LSTM layers, the model would not
    # capture very well the upward and downward trend.
    my_model.add(LSTM(units=50, return_sequences=True, input_shape=(input_variable.shape[1],1), kernel_initializer='uniform'))
    my_model.add(Dropout(0.2))

    # Adding a second LSTM layer and Dropout regularisation
    # No need to specify any input shape here because we have already defined that we have 50 neurons in the
    # previous layer.
    my_model.add(LSTM(units=50,return_sequences=True,  kernel_initializer='uniform'))
    my_model.add(Dropout(0.2))

    # Adding a fourth LSTM layer and Dropout regularisation
    # This is the last LSTM layer that is  added! Thus the return sequences is set to  false.
    my_model.add(LSTM(units = 25, kernel_initializer='uniform'))
    my_model.add(Dropout(0.2))

    # Adding the output layer
    # We are not adding an LSTM layer. We are fully connecting the outward layer to the previous LSTM layer.
    # As a result, we use a DENSE layer to make this full connection.
    my_model.add(Dense(units=1, kernel_initializer='uniform'))

    # Compiling the RNN
    # For RNN and also in the Keras documentation, an RMSprop is recommended.
    # But experimenting with other optimizers, one can also use the adam optimizer.
    # The adam optimizer is actually always a good choice and very powerfull too!
    # In general, the most commonly used optimizers are adam and RMSprop
    optimizer = RMSprop(lr=learning_rate)
    my_model.compile(optimizer=optimizer, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model

def train_model(model, xvalues, yvalues, epochs, batch):

    # Fitting the model to the Training set
    history = model.fit(xvalues, yvalues, epochs=epochs, batch_size=batch)

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist

def plot_the_loss_curve(epochs, difference,string):

    plt.figure(1)
    plt.xlabel("Epoch")
    plt.ylabel(string)
    plt.plot(epochs, difference)
    plt.legend()
    plt.show()

def plot_generation(ax, x_values, y_values, string):

    y_values_dates = create_dates(x_values, y_values)
    ax.plot(y_values_dates, linewidth=0.5)
    ax.set_xlabel("Settlement Periods")
    ax.set_ylabel(string)
    plt.show()

def plot_prediction_zoomed_in(fig, ax, x_values, prediction, true_values):

    y_values_dates = create_dates(x_values, prediction)
    fig.suptitle('LSTM: Single Step Prediction', fontsize=16)
    ax[0].plot(y_values_dates, label="LSTM Prediction")
    ax[0].set_xlabel("Settlement Periods")
    ax[0].set_ylabel("Electricity Load [MW]")
    y_values_dates = create_dates(x_values, true_values)
    ax[0].plot(y_values_dates, label="Actual")
    fig.legend()

    y_values_dates = create_dates(x_values, abs(prediction-true_values))
    ax[1].plot(y_values_dates, label="Absolute Error", color="black")
    ax[1].set_xlabel("Settlement Periods")
    ax[1].set_ylabel("Error in Prediction [MW]")
    fig.legend()

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(round(features_df[i, -1])),
                                   month=int(round(features_df[i, -2])),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=int(((features_df[i, -4] -1) % 2 ) * 30)) for i in range(len(features_df))]

    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates
