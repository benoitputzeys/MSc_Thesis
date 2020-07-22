import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv')
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv')
dates = X.iloc[:,-1]
series = y.iloc[:,-1]/1000

# Decompose the data into daily, weekly and annual seasonal components.
# To this, a residual and a trend is added as well.
daily_components = sm.tsa.seasonal_decompose(series, period=48)
adjusted_nd = series - daily_components.seasonal
weekly_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*7)
adjusted_nd_nw = series - daily_components.seasonal - weekly_components.seasonal
annual_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*365)

# Define the daily and weekly seasonalities.
daily_seasonality = daily_components.seasonal
weekly_seasonality = weekly_components.seasonal

# Compute the settlement periods of the week, not of the day.
settlement_period = X["Settlement Period"]+(48*X["Day of Week"])

# This section might take some time but calculating the mean for each is "safer" this way.
# (If a value is missing in the original data, that is not a problem in computing the mean per week.)
mean_each_week = series.copy()
counter = 0
for i in range(len(X)-1):
    mean_each_week[i-counter:i+1] = np.mean(series[i-counter:i+1])
    counter = counter + 1
    if (X["Day of Week"][i] == 6) & (X["Day of Week"][i+1]==0):
        counter = 0
mean_each_week.iloc[-1]=mean_each_week.iloc[-2]

# The residual is anything that cannot be decomposed into the daily, weekly or mean components.
residual = series - daily_seasonality - weekly_seasonality - mean_each_week

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
# Thus substract the mean of the week.
modified_recons = series- mean_each_week

# Create a dataframe that contains the correct indices (1-336) and the load values.
modified_recons = pd.DataFrame({'SP':settlement_period, 'Load':modified_recons.values})

# Compute the mean and variation for each x.
df_stats = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(337):
    df_stats.iloc[i-1,1]=np.mean(modified_recons[modified_recons["SP"]==i].iloc[:,-1])
    df_stats.iloc[i-1,2]=np.std(modified_recons[modified_recons["SP"]==i].iloc[:,-1])

# Use the "template" above and add the mean of the first week of the test set to it.
ANN_pred = pd.read_csv('Electricity_Generation_Prediction/ANN/Single_Multi_Step_Prediction/Pred_Test.csv')
ANN_pred = ANN_pred/1000

fig1, axs1=plt.subplots(2,1,figsize=(12,6))
axs1[0].plot(dates[-15619:-15283], series[-15619:-15283], color = "blue", label = "Actual Load (True Values)")
axs1[0].plot(dates[-15619:-15283], ANN_pred[:336], color = "orange", label = "ANN Pred")
axs1[0].plot(dates[-15619:-15456], (df_stats.iloc[173:,1]+mean_each_week.iloc[-15619:-15456].values), color = "black", label = "Mean of past loads")
axs1[0].plot(dates[-15456:-15283], (df_stats.iloc[:173,1]+mean_each_week.iloc[-15456:-15283].values), color = "black")
axs1[0].fill_between(dates[-15619:-15456],
                  ((df_stats.iloc[173:,1]-df_stats.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  ((df_stats.iloc[173:,1]+df_stats.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  alpha=0.2, color = "black")
axs1[0].fill_between(dates[-15456:-15283],
                  ((df_stats.iloc[:173,1]-df_stats.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  ((df_stats.iloc[:173,1]+df_stats.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  alpha=0.2, color = "black", label = "+- 1x Stddev")
axs1[0].set_ylabel("Load [GW]", size = 14)

axs1[1].plot(dates[-15619:-15283], abs(ANN_pred.iloc[:336,0].values-series.iloc[-15619:-15283].values), label = "Absolute Error", alpha = 1, color = "red")
axs1[1].set_xlabel('Date',size = 14)
axs1[1].set_ylabel('Absolute Error [GW]',size = 14)

loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs1[1].xaxis.set_major_locator(loc)
axs1[0].xaxis.set_major_locator(loc)
fig1.autofmt_xdate(rotation=10)

axs1[1].legend(loc=(1.04,0.9))
axs1[0].legend(loc=(1.04,0.6))

axs1[0].grid(True)
axs1[1].grid(True)

fig1.show()

# For more clarity, only show the first week in the test set with the historic mean and stddec.
fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates[-15619:-15283], series[-15619:-15283], color = "blue", label = "Test Set (True Values)")
axs2.plot(dates[-15619:-15456], (df_stats.iloc[173:,1]+mean_each_week.iloc[-15619:-15456].values), color = "black", label = "Mean of past loads")
axs2.plot(dates[-15456:-15283], (df_stats.iloc[:173,1]+mean_each_week.iloc[-15456:-15283].values), color = "black")
axs2.fill_between(dates[-15619:-15456],
                  ((df_stats.iloc[173:,1]-df_stats.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  ((df_stats.iloc[173:,1]+df_stats.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  alpha=0.2, color = "black")
axs2.fill_between(dates[-15456:-15283],
                  ((df_stats.iloc[:173,1]-df_stats.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  ((df_stats.iloc[:173,1]+df_stats.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  alpha=0.2, color = "black", label = "+- 1x Stddev")
axs2.set_ylabel("Load [GW]", size = 14)
axs2.set_xlabel("Date", size = 14)

loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=0)
axs2.legend()
axs2.grid(True)

fig2.show()
