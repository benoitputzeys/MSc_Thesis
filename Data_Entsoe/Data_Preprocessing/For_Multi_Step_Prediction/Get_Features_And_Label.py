
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math

# Import the timeseries data and convert the strings to floats
df15 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201501010000-201601010000.csv")
df15['Actual Total Load [MW] - United Kingdom (UK)'] = df15['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

df16 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201601010000-201701010000.csv")
df16['Actual Total Load [MW] - United Kingdom (UK)'] = df16['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

df17 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201701010000-201801010000.csv")
df17['Actual Total Load [MW] - United Kingdom (UK)'] = df17['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

# This csv data has missing values for the first 151 days of the year. (Get rid of them)
df18 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201801010000-201901010000.csv")
df18['Actual Total Load [MW] - United Kingdom (UK)'] = df18['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)
df18 = df18.truncate(before=151 * 48)
df18 = df18.drop([14405])
df18 = df18.drop([14406])

df19 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201901010000-202001010000.csv")
df19['Actual Total Load [MW] - United Kingdom (UK)'] = df19['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)
df19 = df19.drop([14356])
df19 = df19.drop([14357])

# This csv data has missing values for the last 152 days of the year as they lie in the future. (Get rid of them)
df20 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_202001010000-202101010000.csv")
df20.loc[df20['Actual Total Load [MW] - United Kingdom (UK)'] == '-', 'Actual Total Load [MW] - United Kingdom (UK)'] = 0
df20['Actual Total Load [MW] - United Kingdom (UK)'] = df20['Actual Total Load [MW] - United Kingdom (UK)'].astype(float)
df20 = df20.truncate(after=152*48-1)

frames = ([df18, df19, df20])
df = pd.concat(frames)

# Determine the label.
df_label = df["Actual Total Load [MW] - United Kingdom (UK)"]

# Determine the Features.
df_features = pd.DataFrame()
df_features["Total_Load_Past"] = df["Actual Total Load [MW] - United Kingdom (UK)"].shift(+48)

# Create artificial features.
rolling_mean_10 = df_features["Total_Load_Past"].rolling(window=10).mean()
rolling_mean_50 = df_features["Total_Load_Past"].rolling(window=50).mean()
exp_20 = df_features["Total_Load_Past"].ewm(span=20, adjust=False).mean()
exp_50 = df_features["Total_Load_Past"].ewm(span=50, adjust=False).mean()

df_features["Simple_Moving_Average_10_D"] = rolling_mean_10
df_features["Simple_Moving_Average_50_D"] = rolling_mean_50
df_features["Exp_Moving_Average_20_D"] = exp_20
df_features["Exp_Moving_Average_50_D"] = exp_50

# Create the settlement period feature and the day of week feature.
counter = SP = DoW = Day = Month = Year = 0

for i in range(len(df_features)):
    counter = counter + 1
    DoW = np.append(DoW, pd.to_datetime([df.iloc[i, 0][0:10]], format='%d.%m.%Y').weekday.values[0])
    Day = np.append(Day, pd.to_datetime([df.iloc[i, 0][0:10]], format='%d.%m.%Y').day[0])
    Month = np.append(Month, pd.to_datetime([df.iloc[i, 0][0:10]], format='%d.%m.%Y').month[0])
    Year = np.append(Year, pd.to_datetime([df.iloc[i, 0][0:10]], format='%d.%m.%Y').year[0])
    if counter == 49:
        counter = 1
    SP = np.append(SP, counter)

X_inter = np.concatenate((DoW.reshape(len(DoW),1), SP.reshape(len(SP),1), Day.reshape(len(Day),1), Month.reshape(len(Month),1), Year.reshape(len(Year),1)), axis=1)
X_inter = X_inter[1:]
X = np.concatenate((df_features, X_inter), axis=1)
y = df_label.values
y = np.reshape(y, (len(y), 1))

X = X[48:]
y = y[48:]

# After having shifted the data, the nan values have to be replaces in order to have good predictions.
replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
replace_nan.fit(X)
X = replace_nan.transform(X)

replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
replace_nan.fit(y)
y = replace_nan.transform(y)

np.savetxt("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction_Outside_Test_Set/X.csv", X, delimiter=",")
np.savetxt("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction_Outside_Test_Set/y.csv", y, delimiter=",")
#
# plt.plot(X[:,0], label='Electricity Generation 2 SP ago', linewidth=0.5 )
# plt.xlabel("Actual Settlement Period")
# plt.ylabel("Electricity Generation [MW]")
# #plt.plot(y[-48*3:,0], label='Total Generation Actual', linewidth=0.5 )
# plt.plot(X[:,1], label='10 Day MA', color='black' , linewidth=0.5 )
# #plt.plot(X[-48*3:,2], label='50 Day SMA', color='black',  linewidth=0.5 )
# #plt.plot(X[-48*3:,3], label='10 Day Exp MA', color='red',  linewidth=0.5 )
# #plt.plot(X[-48*3:,4], label='50 Day Exp MA', color='red',  linewidth=0.5 )
# plt.legend()
# plt.show()
