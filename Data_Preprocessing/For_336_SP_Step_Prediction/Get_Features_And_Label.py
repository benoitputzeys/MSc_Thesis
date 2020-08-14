import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.ticker as plticker

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
df_features["Load_Past"] = df["Load"].shift(+48*7)
df_features["Transmission_Past"] = df["Transmission"].shift(+48*7)

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

X.to_csv("Data_Preprocessing/For_336_SP_Step_Prediction/X.csv")
y.to_csv("Data_Preprocessing/For_336_SP_Step_Prediction/y.csv")

fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(X.iloc[35000:35000+48*7,0]/1000,color = "blue", label='Training Set')
axs1.set_xlabel("Date", size = 14)
axs1.set_ylabel("Electricity Generation, GW", size = 14)
axs1.plot(X.iloc[35000:35000+48*7,3]/1000, label='48 SP MA', color='green')
axs1.plot(X.iloc[35000:35000+48*7,6]/1000, label='48 SP EMA', color='violet')
axs1.grid(True)
fig1.autofmt_xdate(rotation=0)
axs1.tick_params(axis = "both", labelsize = 12)
axs1.legend(fontsize=12)
fig1.show()
fig1.savefig("Data_Preprocessing/Figures/Explain_Moving_Averages.pdf", bbox_inches='tight')
