import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv')
y = y.set_index("Time")

X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv')
X = X.set_index("Time")
dates = X.iloc[:,-1]
series = y.iloc[:,-1]/1000

# Decompose the data into daily, weekly and annual seasonal components.
# To this, a residual and a trend is added as well.
daily_components = sm.tsa.seasonal_decompose(series, period=48)
adjusted_nd = series - daily_components.seasonal
weekly_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*7)
adjusted_nd_nw = series - daily_components.seasonal - weekly_components.seasonal
annual_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*365)

# Delete values because NaN values come up when computing the trend.
daily_seasonality = daily_components.seasonal
weekly_seasonality = weekly_components.seasonal
# trend = weekly_components.trend
# trend[:168] = weekly_components.trend[169]
# trend[-168:] = weekly_components.trend.iloc[-169]

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

# Plot the individual components but also the actual series.
fig, axs=plt.subplots(4,1,figsize=(12,10))
axs[0].plot(dates[36000:38020],daily_seasonality[36000:38020], color = "blue")
axs[0].set_ylabel("Daily S. [GW]", size = 14)
axs[1].plot(dates[36000:38020],weekly_seasonality[36000:38020], color = "blue")
axs[1].set_ylabel("Weekly S. [GW]", size = 14)
axs[2].plot(dates[36000:38020],mean_each_week[36000:38020], color = "blue")
axs[2].set_ylabel("Weekly Average \n[GW]", size = 14)
axs[3].plot(dates[36000:38020], residual[36000:38020] , color = "blue")
axs[3].set_ylabel("Residual [GW]", size = 14)
axs[3].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*7) # Puts ticks at regular intervals
axs[0].xaxis.set_major_locator(loc), axs[1].xaxis.set_major_locator(loc), axs[2].xaxis.set_major_locator(loc), axs[3].xaxis.set_major_locator(loc)
axs[0].grid(True), axs[1].grid(True), axs[2].grid(True), axs[3].grid(True)
fig.autofmt_xdate(rotation = 7)
axs[3].tick_params(axis = "both", labelsize = 14), axs[2].tick_params(axis = "both", labelsize = 14), axs[1].tick_params(axis = "both", labelsize = 14), axs[0].tick_params(axis = "both", labelsize = 14)
fig.show()
fig.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Decomposition_Zoomed_In.pdf", bbox_inches='tight')

# Plot the weekly average
fig, axs=plt.subplots(1,1,figsize=(15,6))
axs.plot(dates,mean_each_week, color = "blue", linewidth = 1)
axs.set_ylabel("Weekly Average [GW]", size = 18)
axs.set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*130) # this locator puts ticks at regular intervals
axs.grid(True)
axs.xaxis.set_major_locator(loc)
axs.tick_params(axis = "both", labelsize = 14)
fig.autofmt_xdate(rotation = 12)
fig.show()
fig.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Weekly_Average.pdf", bbox_inches='tight')

# # Plot the whole decomposition of the whole actual series.
# fig, axs=plt.subplots(4,1,figsize=(12,10))
# axs[0].plot(daily_seasonality, color = "blue", linewidth = 0.5)
# axs[0].set_ylabel("Daily S. [GW]", size = 14)
# axs[1].plot(weekly_seasonality, color = "blue", linewidth = 0.5)
# axs[1].set_ylabel("Weekly S. [GW]", size = 14)
# axs[2].plot(mean_each_week, color = "blue")
# axs[2].set_ylabel("Weekly Average \n[GW]", size = 14)
# axs[3].plot(dates, residual , color = "blue", linewidth = 0.5)
# axs[3].set_ylabel("Residual [GW]", size = 14)
# axs[3].set_xlabel("Date", size = 18)
# loc = plticker.MultipleLocator(base=48*125) # this locator puts ticks at regular intervals
# axs[0].grid(True)
# axs[1].xaxis.set_major_locator(loc)
# axs[1].grid(True)
# axs[2].xaxis.set_major_locator(loc)
# axs[2].grid(True)
# axs[3].xaxis.set_major_locator(loc)
# axs[3].grid(True)
# fig.autofmt_xdate(rotation = 12)
# axs[0].tick_params(axis = "both", labelsize = 14)
# axs[1].tick_params(axis = "both", labelsize = 14)
# axs[2].tick_params(axis = "both", labelsize = 14)
# axs[3].tick_params(axis = "both", labelsize = 14)
# fig.show()
#
# Christmas?
fig, axs=plt.subplots(4,1,figsize=(12,10))
axs[0].plot(dates[33500:36020],daily_seasonality[33500:36020], color = "blue")
axs[0].set_ylabel("Daily S. [GW]", size = 14)
axs[1].plot(dates[33500:36020],weekly_seasonality[33500:36020], color = "blue")
axs[1].set_ylabel("Weekly S. [GW]", size = 14)
axs[2].plot(dates[33500:36020],mean_each_week[33500:36020], color = "blue")
axs[2].set_ylabel("Weekly Average \n[GW]", size = 14)
axs[3].plot(dates[33500:36020], residual[33500:36020] , color = "blue")
axs[3].set_ylabel("Residual [GW]", size = 14)
axs[3].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*10) # this locator puts ticks at regular intervals
axs[0].xaxis.set_major_locator(loc), axs[1].xaxis.set_major_locator(loc), axs[2].xaxis.set_major_locator(loc), axs[3].xaxis.set_major_locator(loc)
axs[0].grid(True), axs[1].grid(True), axs[2].grid(True), axs[3].grid(True)
axs[0].tick_params(axis = "both", labelsize = 14), axs[1].tick_params(axis = "both", labelsize = 14), axs[2].tick_params(axis = "both", labelsize = 14), axs[3].tick_params(axis = "both", labelsize = 14)
fig.autofmt_xdate(rotation = 5)
fig.show()
fig.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Decomposition_around_Christmas.pdf", bbox_inches='tight')

# To check if the decomposition is correct, the reconstruction should give the initial series.
# In order to calculate the trend, the decomposition requires to set a section of the first and last segment of the series to NaN
# Get rid of these sections to continue the calculations.
reconstruction = daily_seasonality + weekly_seasonality + mean_each_week + residual

# Plot the reconstruction, the actual series and the error between the 2.
fig1, axs1=plt.subplots(3,1,figsize=(12,8))
axs1[0].plot(dates[-48*7*2:-48*7+5],reconstruction[-48*7*2:-48*7+5], color = "blue")
axs1[0].set_ylabel("Modified Timeseries \n[GW]", size = 12)
axs1[1].plot(dates[-48*7*2:-48*7+5],series[-48*7*2:-48*7+5], color = "blue")
axs1[1].set_ylabel("Original Timeseries \n[GW]", size = 12)
axs1[2].plot(dates[-48*7*2:-48*7+5],(reconstruction-series)[-48*7*2:-48*7+5]/1000, color = "blue")
axs1[2].set_ylabel("Error [GW]", size = 12)
axs1[2].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs1[0].xaxis.set_major_locator(loc), axs1[1].xaxis.set_major_locator(loc), axs1[2].xaxis.set_major_locator(loc)
axs1[0].grid(True), axs1[1].grid(True), axs1[2].grid(True)
axs1[0].tick_params(axis = "both", labelsize = 12), axs1[1].tick_params(axis = "both", labelsize = 12), axs1[2].tick_params(axis = "both", labelsize = 12)
fig1.autofmt_xdate(rotation = 7)
fig1.show()

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
modified_timeseries = daily_seasonality + weekly_seasonality + residual

# Plot the modified reconstruction.
fig2, axs2=plt.subplots(3,1,figsize=(12,10))
axs2[0].plot(dates[-48*7-500:-500+5], modified_timeseries[-48*7-500:-500+5], color = "blue")
axs2[0].set_ylabel("Modified Timeseries \n[GW]", size = 14)
axs2[1].plot(dates[-48*7-500:-500+5], series[-48*7-500:-500+5], color = "blue")
axs2[1].set_ylabel("Original Timeseries \n[GW]", size = 14)
axs2[2].plot(dates[-48*7-500:-500+5],(modified_timeseries-series)[-48*7-500:-500+5], color = "blue")
axs2[2].set_ylabel("Error [GW]", size = 14)
axs2[2].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc), axs2[1].xaxis.set_major_locator(loc), axs2[2].xaxis.set_major_locator(loc)
axs2[0].grid(True), axs2[1].grid(True), axs2[2].grid(True)
axs2[0].tick_params(axis = "both", labelsize = 14), axs2[1].tick_params(axis = "both", labelsize = 14), axs2[2].tick_params(axis = "both", labelsize = 14)
fig2.autofmt_xdate(rotation = 9)
fig2.show()
fig2.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Modified_Timeseries_and_Errors.pdf", bbox_inches='tight')

settlement_period = X["Settlement Period"]+(48*X["Day of Week"])

# Create a dataframe that contains the correct indices (1-336) and the load values.
modified_timeseries = pd.DataFrame({'SP':settlement_period, 'Load':modified_timeseries.values})
# Only use the training set data
modified_timeseries_train = modified_timeseries.iloc[31238:62476]

# Plot the projected loads onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(2,1,figsize=(12,10))
axs3[0].plot(dates.iloc[31238+10+48*3:31238+48*7*4+10+48*3+1],
             modified_timeseries_train.iloc[10+48*3:48*7*4+10+48*3+1,-1],
             label = "True Values", alpha = 1, color = "blue")
axs3[0].set_ylabel('Load [GW]',size = 14)
axs3[0].set_xlabel('Date',size = 14)

axs3[1].scatter(modified_timeseries_train["SP"], modified_timeseries_train["Load"], alpha=0.05, label = "Projected Loads", color = "blue")
axs3[1].set_ylabel("Load [GW]", size = 14)
axs3[1].set_xlabel("Settlement Period / Weekday", size = 14)
axs3[1].grid(True)

loc = plticker.MultipleLocator(base=48*7) # Puts ticks at regular intervals
axs3[0].xaxis.set_major_locator(loc)
axs3[0].legend(loc = "upper right",fontsize = 14), axs3[1].legend(loc = "upper right",fontsize = 14)
axs3[1].set_xlabel("Hour / Weekday", size = 14)
loc = plticker.MultipleLocator(base=24) # Puts ticks at regular intervals
plt.xticks(np.arange(1,385, 24), ["00:00 \nMonday", "12:00",
                                  "00:00 \nTuesday", "12:00",
                                  "00:00 \nWednesday", "12:00",
                                  "00:00 \nThursday", "12:00",
                                  "00:00 \nFriday", "12:00",
                                  "00:00 \nSaturday", "12:00",
                                  "00:00 \nSunday","12:00",
                                  "00:00"])
axs3[0].minorticks_on(),axs3[1].minorticks_on(),
axs3[0].grid(True), axs3[1].grid(True)
axs3[1].grid(b=True, which='major'), axs3[1].grid(b=True, which='minor',alpha = 0.2)
axs3[0].tick_params(axis = "both", labelsize = 12), axs3[1].tick_params(axis = "both", labelsize = 12)
fig3.show()
fig3.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Explained_Projected_Load.pdf", bbox_inches='tight')

# Compute the mean and variation for each x.
df_stats = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    df_stats.iloc[i-1,1]=np.mean(modified_timeseries_train[modified_timeseries_train["SP"]==i].iloc[:,-1])
    df_stats.iloc[i-1,2]=np.std(modified_timeseries_train[modified_timeseries_train["SP"]==i].iloc[:,-1])

# Plot the mean and variation for each x.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
axs4.plot(df_stats.iloc[:,0], df_stats.iloc[:,1], color = "blue", label = "Mean of all projected loads")
axs4.fill_between(df_stats.iloc[:,0],
                  (df_stats.iloc[:,1]-df_stats.iloc[:,2]),
                  (df_stats.iloc[:,1]+df_stats.iloc[:,2]),
                  alpha=0.2, color = "blue", label = "+- 1 x Standard Deviation")
axs4.set_ylabel("Load [GW]", size = 14)
axs4.set_xlabel("Hour / Weekday", size = 14)
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
plt.xticks(np.arange(1,385, 24), ["00:00 \nMonday", "12:00",
                                  "00:00 \nTuesday", "12:00",
                                  "00:00 \nWednesday", "12:00",
                                  "00:00 \nThursday", "12:00",
                                  "00:00 \nFriday", "12:00",
                                  "00:00 \nSaturday", "12:00",
                                  "00:00 \nSunday","12:00",
                                  "00:00"])
axs4.legend(fontsize = 12)
axs4.minorticks_on()
axs4.grid(b=True, which='major'), axs4.grid(b=True, which='minor',alpha = 0.2)
axs4.tick_params(axis = "both", labelsize = 12)
axs4.grid(True)
fig4.show()
fig4.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Projected_Load_Mean_and_Stddev.pdf", bbox_inches='tight')

# Plot the mean and variation for each x.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
axs5.fill_between(df_stats.iloc[:,0],
                  (-df_stats.iloc[:,2]),
                  (+df_stats.iloc[:,2]),
                  alpha=0.2, color = "blue", label = "+- 1 x Standard Deviation")
axs5.set_ylabel("Load [GW]", size = 14)
axs5.set_xlabel("Hour / Weekday", size = 14)
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
plt.xticks(np.arange(1,385, 24), ["00:00 \nMonday", "12:00",
                                  "00:00 \nTuesday", "12:00",
                                  "00:00 \nWednesday", "12:00",
                                  "00:00 \nThursday", "12:00",
                                  "00:00 \nFriday", "12:00",
                                  "00:00 \nSaturday", "12:00",
                                  "00:00 \nSunday","12:00"])
axs5.minorticks_on()
axs5.grid(b=True, which='major'), axs5.grid(b=True, which='minor',alpha = 0.2)

axs5.legend(fontsize = 12)
axs5.tick_params(axis = "both", labelsize = 12)
fig5.show()
fig5.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Projected_Load_Stddev.pdf", bbox_inches='tight')

# Use the "template" above and add the mean of the week in question to it.
fig6, axs6=plt.subplots(1,1,figsize=(12,6))
axs6.plot(dates[-48*7*2:-48*7+1], series[-48*7*2:-48*7+1], color = "black", label = "Actual Load")
axs6.plot(settlement_period[-336:],
          (df_stats.iloc[:,1]+mean_each_week.iloc[-336:].values),
          color = "blue", label = "Mean of all projected past loads")
axs6.fill_between(df_stats.iloc[:,0],
                  ((df_stats.iloc[:,1]-df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  ((df_stats.iloc[:,1]+df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  alpha=0.2, color = "blue", label = "+- 1x Standard Deviation")
axs6.set_ylabel("Load [GW]", size = 14)
axs6.set_xlabel("Date", size = 14)
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs6.xaxis.set_major_locator(loc)
axs6.legend(fontsize = 12)
axs6.tick_params(axis = "both", labelsize = 11)
fig6.autofmt_xdate(rotation = 8)
axs6.grid(True)
fig6.show()

# Plot the mean and variation for each x. Together with the location of 2 examples
# that will be explored in more detail.
fig7, axs7=plt.subplots(1,1,figsize=(12,6))
axs7.plot(df_stats.iloc[:,0], df_stats.iloc[:,1], color = "blue", label = "Mean of all projected loads")
axs7.fill_between(df_stats.iloc[:,0],
                  (df_stats.iloc[:,1]-df_stats.iloc[:,2]),
                  (df_stats.iloc[:,1]+df_stats.iloc[:,2]),
                  alpha=0.2, color = "blue", label = "+- 1x Standard Deviation")
axs7.axvline(df_stats.iloc[120,0], linestyle="--", color = "green", label = "Example 1", linewidth = 2)
axs7.axvline(df_stats.iloc[235,0], linestyle="--", color = "orange", label = "Example 2", linewidth = 2)
axs7.set_ylabel("Load [GW]", size = 14)
axs7.set_xlabel("Hour / Weekday", size = 14)
plt.xticks(np.arange(1,385, 24), ["00:00 \nMonday", "12:00",
                                  "00:00 \nTuesday", "12:00",
                                  "00:00 \nWednesday", "12:00",
                                  "00:00 \nThursday", "12:00",
                                  "00:00 \nFriday", "12:00",
                                  "00:00 \nSaturday", "12:00",
                                  "00:00 \nSunday","12:00",
                                  "00:00"])
axs7.legend(fontsize = 12)
axs7.minorticks_on()
axs7.grid(b=True, which='major'), axs7.grid(b=True, which='minor',alpha = 0.2)
axs7.legend(fontsize = 12)
axs7.tick_params(axis = "both", labelsize = 12)
axs7.grid(True)
fig7.show()
fig7.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Projected_Load_Examples_1_&_2.pdf", bbox_inches='tight')

# Define the 2 examples.
example_1 = modified_timeseries[modified_timeseries["SP"]==120]
example_2 = modified_timeseries[modified_timeseries["SP"]==235]

# Plot the histograms of the 2 SPs.
fig8, axs8=plt.subplots(1,2,figsize=(12,6))
axs8[0].hist(example_1.iloc[:,1], bins = 19, color = "green")
axs8[0].set_xlabel("Example 1: Load [GW]", size = 14)
axs8[0].set_ylabel("Count", size = 14)

axs8[1].hist(example_2.iloc[:,1], bins = 19, color = "orange")
axs8[1].set_xlabel("Example 2: Load [GW]", size = 14)
axs8[1].set_ylabel("Count", size = 14)
axs8[0].grid(True)
axs8[1].grid(True)
axs8[0].set_axisbelow(True)
axs8[1].set_axisbelow(True)
fig8.show()
fig8.savefig("Electricity_Generation_Prediction/Historic_Load/Figures/Histograms_Examples_1_2.pdf", bbox_inches='tight')

# Print their mean and standard deviation
print("The mean of example 1 is %.2f" % np.mean(example_1.iloc[:,-1]),"[GW] and the standard deviation is %.2f" % np.std(example_1.iloc[:,-1]),"[GW]." )
print("The mean of example 2 is %.2f" % np.mean(example_2.iloc[:,-1]),"[GW] and the standard deviation is %.2f" % np.std(example_2.iloc[:,-1]),"[GW]." )

df_stats.to_csv("TF_Probability/Results/Projected_Data")

print(np.mean(df_stats.iloc[:,1]))

df_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Training_mean_errors_stddevs.csv")


