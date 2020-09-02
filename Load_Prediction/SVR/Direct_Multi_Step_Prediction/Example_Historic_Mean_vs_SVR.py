import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

########################################################################################################################
# Load the processed data.
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
adjusted_nd = series - daily_components.seasonal # nd means no daily seasonality
weekly_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*7)
adjusted_nd_nw = series - daily_components.seasonal - weekly_components.seasonal # nw means no weekly seasonality

# Define the daily and weekly seasonal trends.
daily_seasonality = daily_components.seasonal
weekly_seasonality = weekly_components.seasonal

# Compute the settlement periods of the week, not of the day. (Going from 1 to 336)
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

# The residual is anything that cannot be decomposed into the daily seasonality, weekly seasonality or weekly mean.
residual = series - daily_seasonality - weekly_seasonality - mean_each_week

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
# Thus substract the mean of the week.
modified_load = series - mean_each_week

# Create a dataframe that contains the correct indices (1-336) and the load values.
modified_timeseries = pd.DataFrame({'SP':settlement_period, 'Load':modified_load.values})

# Compute the mean and variation for each SP.
mean_stddev_per_SP = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(337):
    mean_stddev_per_SP.iloc[i-1,1]=np.mean(modified_timeseries[modified_timeseries["SP"]==i].iloc[:,-1])
    mean_stddev_per_SP.iloc[i-1,2]=np.std(modified_timeseries[modified_timeseries["SP"]==i].iloc[:,-1])

########################################################################################################################
# Load the 7-day ahead prediction of the ANN and plot with the historic mean and variability
########################################################################################################################

# Use the "template" above and add the mean of the first week of the test set to it.
SVR_pred = pd.read_csv('Load_Prediction/SVR/Direct_Multi_Step_Prediction/Pred_Test.csv')
SVR_pred = SVR_pred.iloc[:,-1]

# Plot the variability from the training set in blue with the values from the test set in black and the SVR
# prediction in orange. The specific values have to be found individually because the template can only be used for a
# "whole" week. The week in the test set starts on Thursday  afternoon and ends on Thursday afternoon. This contains 2
# weeks, the first from thursday afternoon to Sunday (SP 173 to 336) and the second from Monday to Thursday afternoon
# (SP 1 to 173).
fig1, axs1=plt.subplots(2,1,figsize=(12,8))
axs1[0].plot(dates[-15619:-15282], series[-15619:-15282], color = "black", label = "Test Set")
axs1[0].plot(dates[-15619:-15282], SVR_pred[:337], color = "orange", label = "SVR Prediction")
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
             (SVR_pred.iloc[:337].values-series.iloc[-15619:-15282].values),
             label = "Error \n(Prediction - True Values)", alpha = 1, color = "red")
axs1[1].set_xlabel('Date',size = 14)
axs1[1].set_ylabel('Error [GW]',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs1[1].xaxis.set_major_locator(loc), axs1[0].xaxis.set_major_locator(loc)
fig1.autofmt_xdate(rotation=10)
axs1[0].tick_params(axis = "both", labelsize = 12), axs1[1].tick_params(axis = "both", labelsize = 12)
axs1[1].legend(loc=(1.04,0.85)), axs1[0].legend(loc=(1.04,0.7))
axs1[0].grid(True), axs1[1].grid(True)
fig1.show()
# Save the figure.
fig1.savefig("Load_Prediction/SVR/Figures/DMSP_Prediction_Compared_to_Historic_Variability.pdf", bbox_inches='tight')

########################################################################################################################
# For more clarity, only show the first week in the test set with the historic mean and stdev.
########################################################################################################################

# For more clarity, only show the first week in the test set with the historic mean and stddev.
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
axs2.set_ylabel("Load, GW", size = 14)
axs2.set_xlabel("Date", size = 14)

loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=10)
axs2.tick_params(axis = "both", labelsize = 12), axs2.tick_params(axis = "both", labelsize = 12)
axs2.legend(fontsize = 12, loc = "upper right")
axs2.grid(True)
fig2.show()

