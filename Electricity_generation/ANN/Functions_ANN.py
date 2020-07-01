
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.layers import Dense
from keras.layers import Dropout
import datetime
from pandas import DataFrame

def plot_the_loss_curve(x_value,metric,string):

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel(string)

    plt.plot(x_value,metric, label="Loss")
    plt.legend()
    plt.show()

def create_model(dim, learning_rate):

    # Create the model.
    my_model = keras.Sequential()

    # Input shape corresponds to the number of columns (the features day, month and year) of the dataframe,
    # excpet for the output label, the temperature.
    my_model.add(Dense(units=50, kernel_initializer='uniform', input_dim = dim, activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=75, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

    opt = keras.optimizers.Adam(lr=learning_rate)
    my_model.compile(optimizer=opt, loss='mae', metrics=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error'])

    return my_model


def train_model(model, xvalues, yvalues, epochs, batch):

    history = model.fit(xvalues, yvalues, batch_size=batch, epochs=epochs)

    # To track the progression of training, gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    return hist

def plot_total_generation(x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    plt.plot(y_values_dates, linewidth=0.5)
    plt.xlabel("Settlement Periods")
    plt.ylabel(string)
    plt.show()

def plot_actual_generation(ax, x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    ax[0].plot(y_values_dates, linewidth=0.5)
    ax[0].set_xlabel("Settlement Periods")
    ax[0].set_ylabel(string)
    plt.show()

def plot_predicted_generation(ax, x_values, y_values, string):

    y_values_dates = create_dates(x_values,y_values)
    ax[1].plot(y_values_dates, linewidth=0.5)
    ax[1].set_xlabel("Settlement Periods")
    ax[1].set_ylabel(string)
    plt.show()

def plot_error(ax,x_values, error, string):

    y_values_dates = create_dates(x_values,error)
    ax[2].plot(y_values_dates, linewidth=0.5)
    ax[2].set_xlabel("Settlement Periods")
    ax[2].set_ylabel(string)
    plt.show()

def plot_prediction_zoomed_in(x_values, y_values, string1):

    y_values_dates = create_dates(x_values, y_values)
    plt.figure(3)
    plt.suptitle('Prediction Zoomed In', fontsize=16)
    plt.xlabel("Settlement Periods")
    plt.ylabel("Predicted Generation")
    plt.plot(y_values_dates, label=string1)
    plt.legend()
    plt.show()

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
