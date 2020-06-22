
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.layers import Dense
from keras.layers import Dropout

def plot_the_loss_curve(x_value,mse):

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(x_value,mse, label="Loss")
    plt.legend()
    plt.show()


def create_model(dim, learning_rate):

    # Create the model.
    my_model = keras.Sequential()

    # Input shape corresponds to the number of columns (the features day, month and year) of the dataframe,
    # excpet for the output label, the temperature.
    my_model.add(Dense(units=25, kernel_initializer='uniform', input_dim = dim, activation='relu'))
    my_model.add(Dropout(0.1))
    my_model.add(Dense(units=75, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.1))
    my_model.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.1))
    my_model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

    opt = keras.optimizers.Adam(lr=learning_rate)
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model


def train_model(model, xvalues, yvalues, epochs, batch):

    #history = model.fit(xvalues, yvalues, batch_size=batch_size, epochs=epochs, shuffle=False)
    history = model.fit(xvalues, yvalues, batch_size=batch, epochs=epochs)

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist

def plot_actual_generation(ax, y_values, string):

    ax[0].plot(y_values, linewidth=0.5)
    ax[0].set_xlabel("Settlement Periods")
    ax[0].set_ylabel(string)
    plt.show()

def plot_predicted_generation(ax,  yvalues, string):

    ax[1].plot( yvalues, linewidth=0.5)
    ax[1].set_xlabel("Settlement Periods")
    ax[1].set_ylabel(string)
    plt.show()

def plot_error(ax, error, string):
    ax[2].plot( error, linewidth=0.5)
    ax[2].set_xlabel("Settlement Periods")
    ax[2].set_ylabel(string)
    plt.show()

def plot_prediction_zoomed_in(yvalues1, yvalues2, yvalues3, string1, string2, string3):
    plt.figure(4)
    plt.suptitle('Prediction Zoomed In', fontsize=16)
    plt.xlabel("Settlement Periods")
    plt.ylabel("Predicted Generation")
    plt.plot(yvalues1, label=string1)
    plt.plot(yvalues2, label=string2)
    plt.plot(yvalues3, label=string3)
    plt.legend()
    plt.show()