# Code here.
# Recurrent Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
# Transform the date in datetime type
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Extract the year, month and day of the date of each observation (1 temp. per day).
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day

# Reset the index.
df.reset_index(inplace = True)

# Split the dataframe in training and testing set.
nmb_rows_df = df.shape[0]
index = int(0.8*nmb_rows_df)
train_df = df[0:index].copy()
test_df = df[index:].copy()

# Define the NN input features: Year, Month, Day and Temperature: Drop the Date columns.
# As the temperature of the nex day will be predicted with the temperature from the previous day, the NN should also
# be trained with the previous day temperatures.
train_shifted_df = train_df.copy()
train_shifted_df['Temp'] = train_shifted_df['Temp'].shift(periods = 1, fill_value = 0)
train_shifted_df = train_shifted_df.drop(columns = ["Date"])
Raw_Input = train_shifted_df.iloc[:, 0:4].values
Raw_Output = train_df[["Temp"]].values

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

X_train = np.zeros((4,len(Raw_Input)-30,30))
y_train = np.zeros((len(Raw_Input)-30,1))
for j in range(4):
    for i in range(30, len(Raw_Input)):
        for k in range(30):
            X_train[j, i - 30, k] = Raw_Input[i-k, j]
            y_train[i - 30] = Raw_Input[i,0]