import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

# Import the timeseries data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

# Transform the date in datetime type
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Extract the year, month and day of the date of each observation (1 temp. per day).
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day


# To make the problem smaller, use only the first 3 rows and see how the NN computes the ouput.
test_df = df[-3:].copy()

# Define the hyperparameters.
learning_rate = 0.0005
epochs = 150
batch_size = 40

# Create the model.
my_model = keras.Sequential()
my_model.add(keras.layers.Dense(50, kernel_initializer = 'uniform', activation = 'relu', input_shape = (4,)))
my_model.add(keras.layers.Dense(50, kernel_initializer = 'uniform', activation = 'relu'))
my_model.add(keras.layers.Dense(1, kernel_initializer = 'uniform', activation = 'relu'))
opt = keras.optimizers.Adam(learning_rate)
my_model.compile(loss = 'mean_squared_error', optimizer = opt)
test_values = test_df.values

A = np.ones((4,50))
B = np.ones((50,))

C = np.ones((50,50))
D = np.ones((50,))

E = np.ones((50,1))
F = np.ones((1,))

my_model.layers[0].set_weights((A,B))
my_model.layers[1].set_weights((C,D))
my_model.layers[2].set_weights((E,F))

predicted_NN_temperatures = my_model.predict(test_values)

# my_model.layers[0].set_weights((A,B*0))
# my_model.layers[1].set_weights((C,D*0))
# predicted_NN_temperatures = my_model.predict(test_values)
# my_model.layers[1].set_weights((C,D))
# my_model.layers[0].set_weights((A,B))
# predicted_NN_temperatures = my_model.predict(test_values)
# my_model.layers[0].set_weights((A,B*2))
# my_model.layers[1].set_weights((C,D*2))
# predicted_NN_temperatures = my_model.predict(test_values)
# my_model.layers[0].set_weights((A,B*0))
# my_model.layers[1].set_weights((C,D))
