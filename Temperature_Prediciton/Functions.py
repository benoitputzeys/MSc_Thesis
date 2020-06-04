import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

def plot_the_loss_curve(epochs, mse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min()*0.95, mse.max() * 1.03])
    plt.show()

print("Defined the plot_the_loss_curve function.")


def create_model(my_learning_rate, my_feature_layer):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(my_feature_layer)

    # Describe the topography of the model by calling the tf.keras.layers.Dense
    # method once for each layer. We've specified the following arguments:
    #   * units specifies the number of nodes in this layer.
    #   * activation specifies the activation function (Rectified Linear Unit).
    #   * name is just a string that can be useful when debugging.

    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=15,
                                    activation='relu',
                                    name='Hidden1'))

    # # Define the second hidden layer with 12 nodes.
    # model.add(tf.keras.layers.Dense(units=12,
    #                                 activation='relu',
    #                                 name='Hidden2'))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def train_model(model, xvalues, yvalues, epochs, batch_size=None):
    """Train the model by feeding it data."""

    history = model.fit(xvalues, yvalues, batch_size=batch_size, epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # To track the progression of training, gather a snapshot
    # of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)

    loss = hist["loss"]
    return epochs, loss

def previousday(df):

    df['Temp']=df['Temp'].shift(periods=1,fill_value=0)
    return df

def nested_sum(lst):
    total = 0  # don't use `sum` as a variable name
    for i in lst:
        if isinstance(i, list):  # checks if `i` is a list
            total += nested_sum(i)
        else:
            total += i
    return total
