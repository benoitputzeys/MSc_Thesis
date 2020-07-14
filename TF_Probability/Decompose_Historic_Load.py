import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv')
dates = y.iloc[:,1]
series = y.iloc[:,-1]

# Decompose the data into daily, weekly and annual seasonal components.
# To this, a residual and a trend is added as well.
daily_components = sm.tsa.seasonal_decompose(series, period=48)
adjusted_nd = series - daily_components.seasonal
weekly_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*7)
adjusted_nd_nw = series - daily_components.seasonal - weekly_components.seasonal
annual_components= sm.tsa.seasonal_decompose(adjusted_nd_nw, period=48*365)
adjusted_nd_nw_na = series - daily_components.seasonal - weekly_components.seasonal - annual_components.seasonal

# Plot the individual components but also the actual series.
fig, axs=plt.subplots(5,1,figsize=(12,10))
axs[0].plot(daily_components.seasonal, color = "blue", linewidth = 0.5)
axs[0].set_ylabel("Daily [MW]", size = 12)
axs[1].plot(weekly_components.seasonal, color = "blue", linewidth = 0.5)
axs[1].set_ylabel("Weekly [MW]", size = 12)
axs[2].plot(annual_components.seasonal, color = "blue", linewidth = 0.5)
axs[2].set_ylabel("Annual [MW]", size = 12)
axs[3].plot(annual_components.trend, color = "blue")
axs[3].set_ylabel("Trend [MW]", size = 12)
axs[4].plot(dates, annual_components.resid, color = "blue", linewidth = 0.5)
axs[4].set_ylabel("Residual [MW]", size = 12)
axs[4].set_xlabel("Date", size = 14)
loc = plticker.MultipleLocator(base=4800) # this locator puts ticks at regular intervals
axs[0].xaxis.set_major_locator(loc)
axs[1].xaxis.set_major_locator(loc)
axs[2].xaxis.set_major_locator(loc)
axs[3].xaxis.set_major_locator(loc)
axs[4].xaxis.set_major_locator(loc)
fig.autofmt_xdate(rotation = 15)
fig.show()

# To check if the decomposition is correct, the reconstruction should give the initial series.
# In order to calculate the trend, the decomposition requires to set a section of the first and last segment of the series to NaN
# Get rid of these sections to continue the calculations.
reconstruction = daily_components.seasonal + weekly_components.seasonal + annual_components.seasonal + annual_components.trend + annual_components.resid

# Plot the reconstruction, the actual series and the error between the 2.
fig1, axs1=plt.subplots(3,1,figsize=(12,8))
axs1[0].plot(dates.iloc[69278-48*7-1:69278], reconstruction.iloc[69278-48*7-1:69278]/1000, color = "blue")
axs1[0].set_ylabel("Reconstruction [GW]", size = 12)
axs1[1].plot(dates.iloc[69278-48*7-1:69278], series[69278-48*7-1:69278]/1000, color = "blue")
axs1[1].set_ylabel("Original [GW]", size = 12)
axs1[2].plot(dates.iloc[69278-48*7-1:69278], (reconstruction-series).iloc[69278-48*7-1:69278]/1000, color = "blue")
axs1[2].set_ylabel("Error [GW]", size = 12)
axs1[2].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs1[0].xaxis.set_major_locator(loc)
axs1[0].grid(True)
axs1[1].xaxis.set_major_locator(loc)
axs1[1].grid(True)
axs1[2].xaxis.set_major_locator(loc)
axs1[2].grid(True)
fig1.autofmt_xdate(rotation = 15)
fig1.show()

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
ds_ws_rs = daily_components.seasonal + weekly_components.seasonal + annual_components.resid
#ds_ws_rs = ds_ws_rs.iloc[8760:69278]

# Plot the modified reconstruction.
fig2, axs2=plt.subplots(3,1,figsize=(12,10))
axs2[0].plot(ds_ws_rs/1000, color = "blue", linewidth = 0.5)
axs2[0].set_ylabel("Modified Reconstruction [GW]", size = 12)
axs2[1].plot(series/1000, color = "blue", linewidth = 0.5)
axs2[1].set_ylabel("Actual [GW]", size = 14)
axs2[2].plot(dates,(ds_ws_rs-series)/1000, color = "blue", linewidth = 0.5)
axs2[2].set_ylabel("Error", size = 14)
axs2[2].set_xlabel("Date", size = 14)
loc = plticker.MultipleLocator(base=4800) # this locator puts ticks at regular intervals
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
df_ds_ws_rs = pd.DataFrame({'SP':settlement_period, 'Load':ds_ws_rs.values})

# Add the mean of the week in question to the reconstructed series.
mean = np.mean(series.iloc[-48*7:])
projected_load = df_ds_ws_rs.iloc[:, 1]

# Plot the projected loads onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(settlement_period, projected_load/1000, alpha=0.05, label = "Projected Loads", color = "blue")
#axs3.plot(settlement_period[-48*7:], series[-48*7:]/1000, color = "red", label = "Load from week in question")
axs3.set_ylabel("Load [GW]", size = 14)
axs3.set_xlabel("Settlement Period", size = 14)
axs3.grid(True)
axs3.legend()
fig3.show()

# Compute the mean and variation for each x.
df_stats = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(337):
    df_stats.iloc[i-1,1]=np.mean(df_ds_ws_rs[df_ds_ws_rs["SP"]==i].iloc[:,-1])
    df_stats.iloc[i-1,2]=np.std(df_ds_ws_rs[df_ds_ws_rs["SP"]==i].iloc[:,-1])

# Plot the mean and variation for each x.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
axs4.plot(df_stats.iloc[:,0], df_stats.iloc[:,1]/1000, color = "blue", label = "Mean of all projected loads")
axs4.fill_between(df_stats.iloc[:,0],  (df_stats.iloc[:,1]-df_stats.iloc[:,2])/1000,  (df_stats.iloc[:,1]+df_stats.iloc[:,2])/1000,alpha=0.2, color = "blue", label = "Stddev")
axs4.set_ylabel("Load [GW]", size = 14)
axs4.set_xlabel("Settlement Period", size = 14)
axs4.legend()
axs4.grid(True)
fig4.show()

# Use the "template" above and add the mean of the week in question to it.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
axs5.plot(settlement_period[-336:], (df_stats.iloc[:,1]+mean)/1000, color = "blue", label = "Mean of all projected loads")
axs5.plot(settlement_period[-336:], series[-48*7:]/1000, color = "red", label = "Actual Load of most recent week")
axs5.fill_between(df_stats.iloc[:,0], ((df_stats.iloc[:,1]-df_stats.iloc[:,2])+mean)/1000, ((df_stats.iloc[:,1]+df_stats.iloc[:,2])+mean)/1000,alpha=0.2, color = "blue", label = "Stddev")
axs5.set_ylabel("Load [GW]", size = 14)
axs5.set_xlabel("Settlement Period", size = 14)
axs5.legend()
axs5.grid(True)
fig5.show()

df_stats.to_csv("TF_Probability/Results/Projected_Data")

print(np.mean(df_stats.iloc[:,1]))
