import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import datetime as dt

df = pd.read_csv("Data_Preprocessing/Load_and_Transmission_Data.csv")
df = df.rename(columns={"Unnamed: 0": "Timestamp"}, errors="raise")

df["Timestamp"] = [dt.datetime.strptime(df.iloc[i,0][0:16], '%Y-%m-%d %H:%M') for i in range(len(df))]
df["Time"] = df["Timestamp"]
df = df.set_index(["Time"])

fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(df.iloc[:,1], color = "blue", linewidth = 0.5)
axs1.set_ylabel("Load in UK, MW", size = 14)
axs1.set_xlabel("Date", size = 14)
axs1.grid(True)
#fig1.suptitle("Electricity Load in the UK from January 2016 to July 2020",fontsize=15)
fig1.show()

# Determine the Features.
df_features = pd.DataFrame()
df_features["Load_Past"] = df["Load"].shift(+1)
df_features["Transmission_Past"] = df["Transmission"].shift(+1)

# Create artificial features.
rolling_mean_10 = df_features["Load_Past"].rolling(window=10).mean()
rolling_mean_48 = df_features["Load_Past"].rolling(window=48).mean()
rolling_mean_336 = df_features["Load_Past"].rolling(window=336).mean()
exp_10 = df_features["Load_Past"].ewm(span=10, adjust=False).mean()
exp_48 = df_features["Load_Past"].ewm(span=48, adjust=False).mean()

df_features["Simple_Moving_Average_10_SP"] = rolling_mean_10
df_features["Simple_Moving_Average_48_SP"] = rolling_mean_48
df_features["Simple_Moving_Average_336_SP"] = rolling_mean_336
df_features["Exp_Moving_Average_10_SP"] = exp_10
df_features["Exp_Moving_Average_48_SP"] = exp_48

# Create date relevant features.
df_features["Settlement Period"] = df['Timestamp'].dt.hour*2+1+df['Timestamp'].dt.minute/30
df_features["Day of Week"] = df['Timestamp'].dt.weekday
df_features['Day'] = df['Timestamp'].dt.day
df_features['Month'] = df['Timestamp'].dt.month
df_features['Year'] = df['Timestamp'].dt.year

df["Timestamp"]  = [pd.to_datetime(df.iloc[i,0]).strftime("%Y:%m:%d %H:%M") for i in range(len(df))]
df_features["Time_At_Delivery"] = df["Timestamp"]

X = df_features
# After having shifted the data, the nan values have to be replaced in order to have good predictions.
replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
replace_nan.fit(X.iloc[:,:-1])
X.iloc[:,:-1] = replace_nan.transform(X.iloc[:,:-1])

y = pd.DataFrame({"Load": df.iloc[:,1]})

X.to_csv("Data_Preprocessing/For_2_SP_Step_Prediction/X.csv")
y.to_csv("Data_Preprocessing/For_2_SP_Step_Prediction/y.csv")

#
# plt.plot(X[:,0], label='Electricity Generation 2 SP ago', linewidth=0.5 )
# plt.xlabel("Actual Settlement Period")
# plt.ylabel("Electricity Generation [MW]")
# #plt.plot(y[-48*3:,0], label='Total Generation Actual', linewidth=0.5 )
# plt.plot(X[:,1], label='10 Day MA', color='black' , linewidth=0.5 )
# #plt.plot(X[-48*3:,2], label='48 SP SMA', color='black',  linewidth=0.5 )
# #plt.plot(X[-48*3:,3], label='10 SP Exp MA', color='red',  linewidth=0.5 )
# #plt.plot(X[-48*3:,4], label='48 SP Exp MA', color='red',  linewidth=0.5 )
# #axs1.grid(True)
# plt.legend()
# plt.show()
