
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.layers import Dense
from keras.layers import Dropout
import datetime
from pandas import DataFrame

def plot_the_loss_curve(x_value,metric):
    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    axs.set_xlabel("Epoch", size = 14)
    axs.set_ylabel("Loss (Mean Absolute Error)", size = 14)
    axs.tick_params(axis="both", labelsize=14)
    axs.plot(x_value, metric, color = "blue")
    fig.show()
    fig.savefig("Electricity_Generation_Prediction/ANN/Figures/ANN_Loss.pdf", bbox_inches='tight')


def create_model(dim, learning_rate):

    # Create the model.
    my_model = keras.Sequential()

    # Input shape corresponds to the number of columns (the features day, month and year) of the dataframe,
    # excpet for the output label, the temperature.
    my_model.add(Dense(units=50, kernel_initializer='uniform', input_dim = dim, activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=25, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

    opt = keras.optimizers.Adam(lr=learning_rate)
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model


def train_model(model, xvalues, yvalues, epochs, batch):

    history = model.fit(xvalues, yvalues, batch_size=batch, epochs=epochs, verbose = 2)

    # To track the progression of training, gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist

def plot_total_generation(x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    plt.plot(y_values_dates, linewidth=0.5, color ="blue")
    plt.xlabel("Settlement Periods", size = 14)
    plt.ylabel(string)
    plt.show()

def plot_actual_generation(ax, x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    ax[0].plot(y_values_dates, color = "blue")
    ax[0].set_ylabel(string)
    ax[0].grid(True)

def plot_predicted_generation(ax, x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    ax[1].plot(y_values_dates, color = "black")
    ax[1].set_ylabel(string)
    ax[1].grid(True)

def plot_error(ax,x_values, error, string):

    y_values_dates = create_dates(x_values,error)
    ax[2].plot(y_values_dates, color = "red")
    ax[2].set_ylabel(string)
    ax[2].set_xlabel("Settlement Periods")
    ax[2].grid(True)

def plot_prediction_zoomed_in(x_values, y_values, string1):

    y_values_dates = create_dates(x_values, y_values)
    plt.suptitle('Prediction Zoomed In', fontsize=16)
    plt.xlabel("Settlement Periods")
    plt.ylabel("Predicted Generation")
    plt.plot(y_values_dates, label=string1, color = "blue")
    plt.grid(True)
    plt.legend()

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(round(features_df[i, -1])),
                                   month=int(round(features_df[i, -2])),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((round(features_df[i, -5]) - 1) / 2),
                                   minute=int(((round(features_df[i, -5]) - 1) % 2) * 30)) for i in range(len(features_df))]

    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates
