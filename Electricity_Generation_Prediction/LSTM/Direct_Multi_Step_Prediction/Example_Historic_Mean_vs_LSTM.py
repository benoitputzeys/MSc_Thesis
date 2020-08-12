import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

########################################################################################################################
# Load the data.
########################################################################################################################

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv')
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv')
dates = X.iloc[:,-1]
series = y.iloc[:,-1]/1000

########################################################################################################################
# Calculate the mean per week and subtract that value from the true load. Project on a single week.
########################################################################################################################

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
modified_load = series - mean_each_week

# Create a dataframe that contains the correct indices (1-336) and the load values.
modified_timeseries = pd.DataFrame({'SP':settlement_period, 'Load':modified_load.values})

# Compute the mean and variation for each x.
mean_stddev_per_SP = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(337):
    mean_stddev_per_SP.iloc[i-1,1]=np.mean(modified_timeseries[modified_timeseries["SP"]==i].iloc[:,-1])
    mean_stddev_per_SP.iloc[i-1,2]=np.std(modified_timeseries[modified_timeseries["SP"]==i].iloc[:,-1])

########################################################################################################################
# Load the 7-day ahead prediction of the ANN and plot with the historic mean and variability
########################################################################################################################

# Use the "template" above and add the mean of the first week of the test set to it.
LSTM_pred = pd.read_csv('Electricity_Generation_Prediction/LSTM/Direct_Multi_Step_Prediction/Pred_Test.csv')
LSTM_pred = LSTM_pred.iloc[:,-1]

fig1, axs1=plt.subplots(2,1,figsize=(12,8))
axs1[0].plot(dates[-15619:-15282], series[-15619:-15282], color = "black", label = "Test Set")
axs1[0].plot(dates[-15619:-15282], LSTM_pred[:337], color = "orange", label = "LSTM Prediction")
axs1[0].plot(dates[-15619:-15456], (mean_stddev_per_SP.iloc[173:,1]+mean_each_week.iloc[-15619:-15456].values), color = "blue", label = "Mean of past loads")
axs1[0].plot(dates[-15456:-15283], (mean_stddev_per_SP.iloc[:173,1]+mean_each_week.iloc[-15456:-15283].values), color = "blue")
axs1[0].fill_between(dates[-15619:-15456],
                  ((mean_stddev_per_SP.iloc[173:,1]-mean_stddev_per_SP.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  ((mean_stddev_per_SP.iloc[173:,1]+mean_stddev_per_SP.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  alpha=0.2, color = "blue")
axs1[0].fill_between(dates[-15456:-15283],
                  ((mean_stddev_per_SP.iloc[:173,1]-mean_stddev_per_SP.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  ((mean_stddev_per_SP.iloc[:173,1]+mean_stddev_per_SP.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  alpha=0.2, color = "blue", label = "+- 1x\nStandard Deviation")
axs1[0].set_ylabel("Load, GW", size = 15)
axs1[0].plot(30,30,label = "Error", color = "red")

axs1[1].plot(dates[-15619:-15282],
             (LSTM_pred.iloc[:337].values-series.iloc[-15619:-15282].values),
             label = "Error \n(Prediction - Training Set)", alpha = 1, color = "red")
axs1[1].set_xlabel('2019',size = 15)
axs1[1].set_ylabel('Error, GW',size = 15)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs1[0].grid(True), axs1[1].grid(True)
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs1[0].xaxis.set_major_locator(loc), axs1[1].xaxis.set_major_locator(loc)
fig1.autofmt_xdate(rotation=0)
axs1[0].legend(loc=(1.02,0.62))
axs1[0].tick_params(axis = "both",labelsize = 12)
axs1[1].tick_params(axis = "both",labelsize = 12)

plt.xticks(np.arange(1,338, 48), ["14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])

fig1.show()
fig1.savefig("Electricity_Generation_Prediction/LSTM/Figures/DMST_Prediction_Compared_to_Historic_Variability.pdf", bbox_inches='tight')

# For more clarity, only show the first week in the test set with the historic mean and stddec.
fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates[-15619:-15282], series[-15619:-15282], color = "black", label = "Test Set")
axs2.plot(dates[-15619:-15456], (mean_stddev_per_SP.iloc[173:,1]+mean_each_week.iloc[-15619:-15456].values), color = "blue", label = "Mean of past loads")
axs2.plot(dates[-15456:-15283], (mean_stddev_per_SP.iloc[:173,1]+mean_each_week.iloc[-15456:-15283].values), color = "blue")
axs2.fill_between(dates[-15619:-15456],
                  ((mean_stddev_per_SP.iloc[173:,1]-mean_stddev_per_SP.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  ((mean_stddev_per_SP.iloc[173:,1]+mean_stddev_per_SP.iloc[173:,2])+mean_each_week.iloc[-15619:-15456].values),
                  alpha=0.2, color = "blue")
axs2.fill_between(dates[-15456:-15283],
                  ((mean_stddev_per_SP.iloc[:173,1]-mean_stddev_per_SP.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  ((mean_stddev_per_SP.iloc[:173,1]+mean_stddev_per_SP.iloc[:173,2])+mean_each_week.iloc[-15456:-15283].values),
                  alpha=0.2, color = "blue", label = "+- 1x Standard Deviation")
axs2.set_ylabel("Load, GW", size = 15)
axs2.set_xlabel("2019", size = 15)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2.grid(True),
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc),
fig2.autofmt_xdate(rotation=0)
axs2.legend(loc = "upper right", fontsize = 12)

plt.xticks(np.arange(1,338, 48), ["14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])
axs2.tick_params(axis = "both",labelsize = 12)
fig2.show()

fig2.savefig("Electricity_Generation_Prediction/LSTM/Figures/Actual_Values_Compared_to_Historic_Variability.pdf", bbox_inches='tight')

