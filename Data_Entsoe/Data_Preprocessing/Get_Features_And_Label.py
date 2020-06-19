
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math

# Import the timeseries data and convert the strings to floats
df15 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201501010000-201601010000.csv")
df15['Actual Total Load [MW] - United Kingdom (UK)'] = df15['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

df16 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201601010000-201701010000.csv")
df16['Actual Total Load [MW] - United Kingdom (UK)'] = df16['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

df17 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201701010000-201801010000.csv")
df17['Actual Total Load [MW] - United Kingdom (UK)'] = df17['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

# This csv data has missing values for the first 151 days of the year. (Get rid of them)
df18 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201801010000-201901010000.csv")
df18['Actual Total Load [MW] - United Kingdom (UK)'] = df18['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)
df18 = df18.truncate(before=151 * 48)

df19 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201901010000-202001010000.csv")
df19['Actual Total Load [MW] - United Kingdom (UK)'] = df19['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

# This csv data has missing values for the last 152 days of the year as they lie in the future. (Get rid of them)
df20 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_202001010000-202101010000.csv")
df20.loc[df20['Actual Total Load [MW] - United Kingdom (UK)'] == '-', 'Actual Total Load [MW] - United Kingdom (UK)'] = 0
df20['Actual Total Load [MW] - United Kingdom (UK)'] = df20['Actual Total Load [MW] - United Kingdom (UK)'].astype(float)
df20 = df20.truncate(after=152*48-1)

frames = ([df18, df19, df20])
df = pd.concat(frames)

# Determine the label.
df_label = df["Actual Total Load [MW] - United Kingdom (UK)"]

# Determine the Features.
df_features = pd.DataFrame()
df_features["Total_Load_Past"] = df["Actual Total Load [MW] - United Kingdom (UK)"].shift(+2)

# Create artificial features.
rolling_mean_10 = df["Actual Total Load [MW] - United Kingdom (UK)"].rolling(window=10).mean()
rolling_mean_50 = df["Actual Total Load [MW] - United Kingdom (UK)"].rolling(window=50).mean()
exp_20 = df["Actual Total Load [MW] - United Kingdom (UK)"].ewm(span=20, adjust=False).mean()
exp_50 = df["Actual Total Load [MW] - United Kingdom (UK)"].ewm(span=50, adjust=False).mean()

df_features["Simple_Moving_Average_10_D"] = rolling_mean_10
df_features["Simple_Moving_Average_50_D"] = rolling_mean_50
df_features["Exp_Moving_Average_20_D"] = exp_20
df_features["Exp_Moving_Average_50_D"] = exp_50

df_features["Day_of_Week"] = df["Time (CET)"]

# Create the settlement period feature and the day of week feature.
counter = 0
SP = 0
DoW = 0
for i in range(len(df_features)):
    counter = counter + 1
    DoW = np.append(DoW, pd.to_datetime([df_features.iloc[i, 5][0:10]], format='%d.%m.%Y').weekday.values[0])
    if counter == 49:
        counter = 1
    SP = np.append(SP, counter)

SP = np.delete(SP,0)
DoW = np.delete(DoW,0)

df_features["Settlement_Period"] = SP
df_features["Day_of_Week"] = DoW


# Create your input variable
X = df_features.values
y = df_label.values

y = np.reshape(y, (len(y), 1))

# After having shifted the data, the nan values have to be replaces in order to have good predictions.
replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
replace_nan.fit(X)
X = replace_nan.transform(X)

replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
replace_nan.fit(y)
y = replace_nan.transform(y)

np.savetxt("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Data_Preprocessing/X.csv", X, delimiter=",")
np.savetxt("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data_Entsoe/Data_Preprocessing/y.csv", y, delimiter=",")

# plt.plot(X[-120:,0], label='Total Generation Past', linewidth=0.5 )
# #plt.plot(y[:,0], label='Total Generation Actual', linewidth=0.5 )
# plt.plot(X[-120:,1], label='SMA 10 Day SMA', color='orange' , linewidth=0.5 )
# plt.plot(X[-120:,2], label='SMA 50 Day SMA', color='magenta',  linewidth=0.5 )
# plt.plot(X[-120:,3], label='EXP 10 Day SMA', color='red',  linewidth=0.5 )
# plt.plot(X[-120:,4], label='ECP 50 Day SMA', color='black',  linewidth=0.5 )
  # plt.show()


