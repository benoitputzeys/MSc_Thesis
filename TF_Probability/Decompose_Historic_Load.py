import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv')
dates = y.iloc[:,1]
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

X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
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
axs[0].plot(dates[36000:38000],daily_seasonality[36000:38000], color = "blue")
axs[0].set_ylabel("Daily S. [GW]", size = 12)
axs[1].plot(dates[36000:38000],weekly_seasonality[36000:38000], color = "blue")
axs[1].set_ylabel("Weekly S. [GW]", size = 12)
axs[2].plot(dates[36000:38000],mean_each_week[36000:38000], color = "blue")
axs[2].set_ylabel("Weekly Average [GW]", size = 12)
axs[3].plot(dates[36000:38000], residual[36000:38000] , color = "blue")
axs[3].set_ylabel("Residual [GW]", size = 12)
axs[3].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*7-3) # this locator puts ticks at regular intervals
axs[0].xaxis.set_major_locator(loc)
axs[0].grid(True)
axs[1].xaxis.set_major_locator(loc)
axs[1].grid(True)
axs[2].xaxis.set_major_locator(loc)
axs[2].grid(True)
axs[3].xaxis.set_major_locator(loc)
axs[3].grid(True)
fig.autofmt_xdate(rotation = 10)
fig.show()

# # Plot the whole decomposition of the whole actual series.
# fig, axs=plt.subplots(4,1,figsize=(12,10))
# axs[0].plot(daily_seasonality, color = "blue", linewidth = 0.5)
# axs[0].set_ylabel("Daily S. [GW]", size = 12)
# axs[1].plot(weekly_seasonality, color = "blue", linewidth = 0.5)
# axs[1].set_ylabel("Weekly S. [GW]", size = 12)
# axs[2].plot(mean_each_week, color = "blue")
# axs[2].set_ylabel("Weekly Average [GW]", size = 12)
# axs[3].plot(dates, residual , color = "blue", linewidth = 0.5)
# axs[3].set_ylabel("Residual [GW]", size = 12)
# axs[3].set_xlabel("Date", size = 18)
# loc = plticker.MultipleLocator(base=48*125) # this locator puts ticks at regular intervals
# axs[0].grid(True)
# axs[1].xaxis.set_major_locator(loc)
# axs[1].grid(True)
# axs[2].xaxis.set_major_locator(loc)
# axs[2].grid(True)
# axs[3].xaxis.set_major_locator(loc)
# axs[3].grid(True)
# fig.autofmt_xdate(rotation = 15)
# fig.show()

# # Christmas?
# fig, axs=plt.subplots(4,1,figsize=(12,10))
# axs[0].plot(dates[33500:36000],daily_seasonality[33500:36000], color = "blue")
# axs[0].set_ylabel("Daily S. [GW]", size = 12)
# axs[1].plot(dates[33500:36000],weekly_seasonality[33500:36000], color = "blue")
# axs[1].set_ylabel("Weekly S. [GW]", size = 12)
# axs[2].plot(dates[33500:36000],mean_each_week[33500:36000], color = "blue")
# axs[2].set_ylabel("Weekly Average [GW]", size = 12)
# axs[3].plot(dates[33500:36000], residual[33500:36000] , color = "blue")
# axs[3].set_ylabel("Residual [GW]", size = 12)
# axs[3].set_xlabel("Date", size = 18)
# loc = plticker.MultipleLocator(base=48*10-1) # this locator puts ticks at regular intervals
# axs[0].xaxis.set_major_locator(loc)
# axs[0].grid(True)
# axs[1].xaxis.set_major_locator(loc)
# axs[1].grid(True)
# axs[2].xaxis.set_major_locator(loc)
# axs[2].grid(True)
# axs[3].xaxis.set_major_locator(loc)
# axs[3].grid(True)
# fig.autofmt_xdate(rotation = 10)
# fig.show()

# To check if the decomposition is correct, the reconstruction should give the initial series.
# In order to calculate the trend, the decomposition requires to set a section of the first and last segment of the series to NaN
# Get rid of these sections to continue the calculations.
reconstruction = daily_seasonality + weekly_seasonality + mean_each_week + residual

# Plot the reconstruction, the actual series and the error between the 2.
fig1, axs1=plt.subplots(3,1,figsize=(12,8))
axs1[0].plot(dates[-48*7:],reconstruction[-48*7:], color = "blue")
axs1[0].set_ylabel("Reconstruction [GW]", size = 12)
axs1[1].plot(dates[-48*7:],series[-48*7:], color = "blue")
axs1[1].set_ylabel("Original [GW]", size = 12)
axs1[2].plot(dates[-48*7:],(reconstruction-series)[-48*7:]/1000, color = "blue")
axs1[2].set_ylabel("Error [GW]", size = 12)
axs1[2].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs1[0].xaxis.set_major_locator(loc)
axs1[0].grid(True)
axs1[1].xaxis.set_major_locator(loc)
axs1[1].grid(True)
axs1[2].xaxis.set_major_locator(loc)
axs1[2].grid(True)
fig1.autofmt_xdate(rotation = 15)
fig1.show()

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
modified_recons = daily_seasonality + weekly_seasonality + residual

# Plot the modified reconstruction.
fig2, axs2=plt.subplots(3,1,figsize=(12,10))
axs2[0].plot(dates[-48*7-500:-500], modified_recons[-48*7-500:-500], color = "blue")
axs2[0].set_ylabel("Modified Reconstruction [GW]", size = 12)
axs2[1].plot(dates[-48*7-500:-500], series[-48*7-500:-500], color = "blue")
axs2[1].set_ylabel("Actual [GW]", size = 14)
axs2[2].plot(dates[-48*7-500:-500],(modified_recons-series)[-48*7-500:-500], color = "blue")
axs2[2].set_ylabel("Error [GW]", size = 14)
axs2[2].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)
axs2[0].grid(True)
axs2[1].xaxis.set_major_locator(loc)
axs2[1].grid(True)
axs2[2].xaxis.set_major_locator(loc)
axs2[2].grid(True)
fig2.autofmt_xdate(rotation = 15)
fig2.show()

X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
settlement_period = X["Settlement Period"]+(48*X["Day of Week"])

# Create a dataframe that contains the correct indices (1-336) and the load values.
modified_recons = pd.DataFrame({'SP':settlement_period, 'Load':modified_recons.values})

# Plot the projected loads onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(modified_recons["SP"], modified_recons["Load"], alpha=0.05, label = "Projected Loads", color = "blue")
#axs3.plot(settlement_period[-48*7:], series[-48*7:]/1000, color = "red", label = "Load from week in question")
axs3.set_ylabel("Load [GW]", size = 14)
axs3.set_xlabel("Settlement Period", size = 14)
axs3.grid(True)
axs3.legend()
fig3.show()

# Compute the mean and variation for each x.
df_stats = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(337):
    df_stats.iloc[i-1,1]=np.mean(modified_recons[modified_recons["SP"]==i].iloc[:,-1])
    df_stats.iloc[i-1,2]=np.std(modified_recons[modified_recons["SP"]==i].iloc[:,-1])

# Plot the mean and variation for each x.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
axs4.plot(df_stats.iloc[:,0], df_stats.iloc[:,1], color = "blue", label = "Mean of all projected loads")
axs4.fill_between(df_stats.iloc[:,0],  (df_stats.iloc[:,1]-df_stats.iloc[:,2]),  (df_stats.iloc[:,1]+df_stats.iloc[:,2]),alpha=0.2, color = "blue", label = "Stddev")
axs4.set_ylabel("Load [GW]", size = 14)
axs4.set_xlabel("Settlement Period", size = 14)
axs4.legend()
axs4.grid(True)
fig4.show()

# Use the "template" above and add the mean of the week in question to it.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
axs5.plot(settlement_period[-336:], (df_stats.iloc[:,1]+mean_each_week.iloc[-336:].values), color = "blue", label = "Mean of all projected loads")
axs5.plot(settlement_period[-336:], series[-48*7:], color = "red", label = "Actual Load of most recent week")
axs5.fill_between(df_stats.iloc[:,0],
                  ((df_stats.iloc[:,1]-df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  ((df_stats.iloc[:,1]+df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  alpha=0.2, color = "blue", label = "Stddev")
axs5.set_ylabel("Load [GW]", size = 14)
axs5.set_xlabel("Settlement Period", size = 14)
axs5.legend()
axs5.grid(True)
fig5.show()

# Plot the mean and variation for each x. Together with the location of 2 examples
# that will be explored in more detail.
fig6, axs6=plt.subplots(1,1,figsize=(12,6))
axs6.plot(df_stats.iloc[:,0], df_stats.iloc[:,1], color = "blue", label = "Mean of all projected loads")
axs6.fill_between(df_stats.iloc[:,0],  (df_stats.iloc[:,1]-df_stats.iloc[:,2]),  (df_stats.iloc[:,1]+df_stats.iloc[:,2]),alpha=0.2, color = "blue", label = "Stddev")
axs6.axvline(df_stats.iloc[120,0], linestyle="--", color = "green", label = "Example 1", linewidth = 2)
axs6.axvline(df_stats.iloc[235,0], linestyle="--", color = "orange", label = "Example 2", linewidth = 2)
axs6.set_ylabel("Load [GW]", size = 14)
axs6.set_xlabel("Settlement Period", size = 14)
axs6.set_xticks([0,50,100,120,150,200,235,250,300,350])
axs6.legend()
axs6.grid(True)
fig6.show()

# Define the 2 examples.
example_1 = modified_recons[modified_recons["SP"]==120]
example_2 = modified_recons[modified_recons["SP"]==235]

# Plot the histograms of the 2 SPs.
fig7, axs7=plt.subplots(1,2,figsize=(12,6))
axs7[0].hist(example_1.iloc[:,1], bins = 21, color = "green")
axs7[0].set_xlabel("Example 1: Load [GW]", size = 14)
axs7[0].set_ylabel("Count", size = 14)

axs7[1].hist(example_2.iloc[:,1], bins = 22, color = "orange")
axs7[1].set_xlabel("Example 2: Load [GW]", size = 14)
axs7[1].set_ylabel("Count", size = 14)
fig7.show()

# Print their mean and standard deviation
print("The mean of example 1 is %.2f" % df_stats.iloc[119,1],"[GW] and the standard deviation is %.2f" % df_stats.iloc[119,2],"[GW]." )
print("The mean of example 2 is %.2f" % df_stats.iloc[234,1],"[GW] and the standard deviation is %.2f" % df_stats.iloc[234,2],"[GW]." )

df_stats.to_csv("TF_Probability/Results/Projected_Data")

print(np.mean(df_stats.iloc[:,1]))
