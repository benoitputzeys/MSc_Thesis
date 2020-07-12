import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv')
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
fig, axs=plt.subplots(5,1,figsize=(12,6))
axs[0].plot(daily_components.seasonal)
axs[0].set_ylabel("Daily", size = 10)
axs[1].plot(weekly_components.seasonal)
axs[1].set_ylabel("Weekly", size = 10)
axs[2].plot(annual_components.seasonal)
axs[2].set_ylabel("Annual", size = 10)
axs[3].plot(annual_components.trend)
axs[3].set_ylabel("Trend", size = 10)
axs[4].plot(series)
axs[4].set_ylabel("Original", size = 10)
axs[4].set_xlabel("Total SPs", size = 14)
fig.show()

# To check if the decomposition is correct, the reconstruction should give the initial series.
# In order to calculate the trend, the decomposition requires to set a section of the first and last segment of the series to NaN
# Get rid of these sections to continue the calculations.
reconstruction = daily_components.seasonal + weekly_components.seasonal + annual_components.seasonal + annual_components.trend + annual_components.resid
reconstruction = reconstruction.iloc[8760:69278]
series = series.iloc[8760:69278]

# Plot the reconstruction, the actual series and the error between the 2.
fig1, axs1=plt.subplots(3,1,figsize=(12,6))
axs1[0].plot(reconstruction.iloc[-48*7:])
axs1[0].set_ylabel("Reconstruction", size = 14)
axs1[1].plot(series.iloc[-48*7:])
axs1[1].set_ylabel("Original", size = 14)
axs1[2].plot((reconstruction-series).iloc[-48*7:])
axs1[2].set_ylabel("Error", size = 14)
axs1[2].set_xlabel("Date", size = 14)
fig1.show()

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
ds_ws_rs = daily_components.seasonal + weekly_components.seasonal + annual_components.resid
ds_ws_rs = ds_ws_rs.iloc[8760:69278]

# Plot the modified reconstruction.
fig2, axs2=plt.subplots(3,1,figsize=(12,6))
axs2[0].plot(ds_ws_rs)
axs2[0].set_ylabel("Modified Reconstruction", size = 12)
axs2[1].plot(series)
axs2[1].set_ylabel("Actual", size = 14)
axs2[2].plot(ds_ws_rs-series)
axs2[2].set_ylabel("Error", size = 14)
axs2[2].set_xlabel("Date", size = 14)
fig2.show()

# Create a dataframe that contains the correct indices (1-336) and the load values.
df_ds_ws_rs = pd.DataFrame({'Index':ds_ws_rs.index, 'Load':ds_ws_rs.values})
j=1
for i in range(len(ds_ws_rs)):
    df_ds_ws_rs["Index"][i] = j
    j = j+1
    if j==48*7+1:
        j=1

# Add the mean of the week in question to the reconstructed series.
mean = np.mean(series.iloc[-48*7])
projected_load = df_ds_ws_rs.iloc[:, 1] + mean

# Plot the projected loads onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(df_ds_ws_rs.iloc[298:60442, 0], projected_load.iloc[298:60442]/1000, alpha=0.1, label = "Projected Loads")
axs3.plot(df_ds_ws_rs.iloc[-48*7+60480: 60480, 0], series[-48*7+60480: 60480]/1000, color = "red", label = "Load from week in question")
axs3.set_ylabel("Load [GW]", size = 10)
axs3.set_xlabel("Date", size = 14)
axs3.legend()
fig3.show()
