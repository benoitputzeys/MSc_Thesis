import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker

########################################################################################################################
# Get the data.
########################################################################################################################

# Read the processed data.
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv')
y = y.set_index("Time")

X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv')
X = X.set_index("Time")
dates = X.iloc[:,-1]
series = y.iloc[:,-1]/1000

########################################################################################################################
# Decompose the data, plot the mean for each week and plot the decomposition around Christmas.
########################################################################################################################

# Decompose the data into daily, weekly and annual seasonal components.
daily_components = sm.tsa.seasonal_decompose(series, period=48)
adjusted_nd = series - daily_components.seasonal
weekly_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*7)
adjusted_nd_nw = series - daily_components.seasonal - weekly_components.seasonal
annual_components= sm.tsa.seasonal_decompose(adjusted_nd, period=48*365)

# Save the weekly and seasonal trends.
daily_seasonality = daily_components.seasonal
weekly_seasonality = weekly_components.seasonal

settlement_period = X["Settlement Period"]+(48*X["Day of Week"])

# Compute the mean for each week
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

# Plot the weekly average
fig, axs=plt.subplots(1,1,figsize=(15,6))
axs.plot(dates,mean_each_week, color = "blue", linewidth = 1)
axs.set_ylabel("Weekly Average, GW", size = 18)
axs.set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*130) # this locator puts ticks at regular intervals
axs.grid(True)
axs.xaxis.set_major_locator(loc)
axs.tick_params(axis = "both", labelsize = 14)
fig.autofmt_xdate(rotation = 0)
plt.xticks(np.arange(1,len(dates), 48*130), ["2016-01","2016-05","2016-09",
                                             "2017-01","2017-06","2017-11",
                                             "2018-02","2018-06","2018-11",
                                             "2019-03","2019-07","2019-12","2020-04"])
fig.show()
fig.savefig("Load_Prediction/Historic_Load/Figures/Weekly_Average.pdf", bbox_inches='tight')

# Decomposition around Christmas
fig, axs=plt.subplots(4,1,figsize=(12,10))
axs[0].plot(dates[33500:36020],daily_seasonality[33500:36020], color = "blue")
axs[0].set_ylabel("Daily S., GW", size = 14)
axs[1].plot(dates[33500:36020],weekly_seasonality[33500:36020], color = "blue")
axs[1].set_ylabel("Weekly S., GW", size = 14)
axs[2].plot(dates[33500:36020],mean_each_week[33500:36020], color = "blue")
axs[2].set_ylabel("Weekly Avg., GW\n", size = 14)
axs[3].plot(dates[33500:36020], residual[33500:36020] , color = "blue")
axs[3].set_ylabel("Residual, GW", size = 14)
axs[3].set_xlabel("Date", size = 18)
loc = plticker.MultipleLocator(base=48*7) # this locator puts ticks at regular intervals
axs[0].xaxis.set_major_locator(loc), axs[1].xaxis.set_major_locator(loc), axs[2].xaxis.set_major_locator(loc), axs[3].xaxis.set_major_locator(loc)
axs[0].grid(True), axs[1].grid(True), axs[2].grid(True), axs[3].grid(True)
axs[0].tick_params(axis = "both", labelsize = 14), axs[1].tick_params(axis = "both", labelsize = 14), axs[2].tick_params(axis = "both", labelsize = 14), axs[3].tick_params(axis = "both", labelsize = 14)
fig.autofmt_xdate(rotation = 0)
plt.xticks(np.arange(1,2520, 48*7), ["2017/11/28",
                                     "12/05",
                                     "12/12",
                                     "12/19","12/26",
                                     "2018/01/02",
                                     "01/09",
                                     "01/16"])
fig.show()
fig.savefig("Load_Prediction/Historic_Load/Figures/Decomposition_around_Christmas.pdf", bbox_inches='tight')

########################################################################################################################
# Subtract the mean of each week from the data thus creating a new timeseries. Create a plot to explain this.
########################################################################################################################

# Create a modified reconstruction of the series where only the daily, weekly and residual components are considered.
modified_timeseries = daily_seasonality + weekly_seasonality + residual

# Create a dataframe that contains the indices (1-336) and the load values.
modified_timeseries = pd.DataFrame({'SP':settlement_period, 'Load':modified_timeseries.values})

# Use the training set data
modified_timeseries_train = modified_timeseries.iloc[31238:62476]

# Plot the projected loads onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(2,1,figsize=(12,10))
axs3[0].plot(dates.iloc[31238+10+48*3:31238+48*7*3+10+48*3+1],
             modified_timeseries_train.iloc[10+48*3:48*7*3+10+48*3+1,-1],
            alpha = 1, color = "blue")
axs3[0].set_ylabel('Electricity Load Training Set, GW',size = 14)

axs3[1].scatter(modified_timeseries_train["SP"].iloc[:3000], modified_timeseries_train["Load"].iloc[:3000], alpha=0.2, label = "Projected Loads", color = "blue")
axs3[1].set_ylabel("Electricity Load Training Set, GW", size = 14)
axs3[1].grid(True)

loc = plticker.MultipleLocator(base=48*7) # Puts ticks at regular intervals
axs3[0].xaxis.set_major_locator(loc)
axs3[1].legend(loc = "upper right",fontsize = 14)
axs3[0].set_xticklabels(["2017/10/16",
                         "2017/10/16",
                         "2017/10/23",
                         "2017/10/30",
                         "2017/11/06"])
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
axs3[0].set_axisbelow(True), axs3[1].set_axisbelow(True)
axs3[1].grid(b=True, which='major'), axs3[1].grid(b=True, which='minor',alpha = 0.2)
axs3[0].tick_params(axis = "both", labelsize = 12), axs3[1].tick_params(axis = "both", labelsize = 12)
fig3.show()
fig3.savefig("Load_Prediction/Historic_Load/Figures/Explained_Projected_Load_3_Weeks.pdf", bbox_inches='tight')

########################################################################################################################
# Use the scatter plot to compute the mean and standard deviation for each week.
########################################################################################################################

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
axs4.set_ylabel("Electricity Load Training Set, GW", size = 14)
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
axs4.set_axisbelow(True)
fig4.show()
fig4.savefig("Load_Prediction/Historic_Load/Figures/Projected_Load_Mean_and_Stddev.pdf", bbox_inches='tight')

########################################################################################################################
# Plot the standard deviation alone for each SP.
########################################################################################################################

zeros = np.zeros((336,))
# Plot the mean and variation for each x.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
axs5.fill_between(df_stats.iloc[:,0],
                  (zeros),
                  (+df_stats.iloc[:,2]),
                  alpha=0.2, color = "blue", label = "Standard deviation in the training set")
axs5.set_ylabel("Electricity Load Training Set, GW", size = 14)
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
axs5.set_axisbelow(True)
fig5.show()
fig5.savefig("Load_Prediction/Historic_Load/Figures/Projected_Load_Stddev.pdf", bbox_inches='tight')

########################################################################################################################
# Use the "template" and add the mean of a week in question to it.
########################################################################################################################

fig6, axs6=plt.subplots(1,1,figsize=(12,6))
axs6.plot(dates[-48*7*2:-48*7+1], series[-48*7*2:-48*7+1], color = "black", label = "Actual Load")
axs6.plot(settlement_period[-336:],
          (df_stats.iloc[:,1]+mean_each_week.iloc[-336:].values),
          color = "blue", label = "Mean of all projected past loads")
axs6.fill_between(df_stats.iloc[:,0],
                  ((df_stats.iloc[:,1]-df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  ((df_stats.iloc[:,1]+df_stats.iloc[:,2])+mean_each_week.iloc[-336:].values),
                  alpha=0.2, color = "blue", label = "+- 1x Standard Deviation")
axs6.set_ylabel("Electricity Load Training Set, GW", size = 14)
axs6.set_xlabel("Date", size = 14)
loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs6.xaxis.set_major_locator(loc)
axs6.legend(fontsize = 12)
axs6.tick_params(axis = "both", labelsize = 11)
fig6.autofmt_xdate(rotation = 8)
axs6.grid(True)
fig6.show()

########################################################################################################################
# Use the "template" to show the probability distribution of 2 SPs. Plot the histograms too!
########################################################################################################################

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
axs7.set_ylabel("Electricity Load Training Set, GW", size = 14)
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
axs7.set_axisbelow(True)
fig7.show()
fig7.savefig("Load_Prediction/Historic_Load/Figures/Projected_Load_Examples_1_&_2.pdf", bbox_inches='tight')

# Define the 2 examples.
example_1 = modified_timeseries[modified_timeseries["SP"]==120]
example_2 = modified_timeseries[modified_timeseries["SP"]==235]

# Plot the histograms of the 2 SPs.
fig8, axs8=plt.subplots(1,2,figsize=(12,6))
axs8[0].hist(example_1.iloc[:,1], bins = 19, color = "green")
axs8[0].set_xlabel("Example 1: Load, GW", size = 14)
axs8[0].set_ylabel("Number of SPs", size = 14)

axs8[1].hist(example_2.iloc[:,1], bins = 19, color = "orange")
axs8[1].set_xlabel("Example 2: Load, GW", size = 14)
axs8[1].set_ylabel("Number of SPs", size = 14)
axs8[0].grid(True)
axs8[1].grid(True)
axs8[0].set_axisbelow(True)
axs8[1].set_axisbelow(True)
fig8.show()
fig8.savefig("Load_Prediction/Historic_Load/Figures/Histograms_Examples_1_2.pdf", bbox_inches='tight')

mean_ex_1 = np.mean(example_1.iloc[:,-1])
mean_ex_2 = np.mean(example_2.iloc[:,-1])
stddev_ex_1 = np.std(example_1.iloc[:,-1])
stddev_ex_2 = np.std(example_2.iloc[:,-1])

# Print their mean and standard deviation
print("The mean of example 1 is %.2f" % mean_ex_1,"GW and the standard deviation is %.2f" % stddev_ex_1,"GW." )
print("The mean of example 2 is %.2f" % mean_ex_2,"GW and the standard deviation is %.2f" % stddev_ex_2,"GW." )

ex_1_num_in_region_1_stev = len(example_1[((mean_ex_1-stddev_ex_1)<example_1.iloc[:,-1]) & (example_1.iloc[:,-1] <(mean_ex_1+stddev_ex_1))])
ex_1_num_in_region_2_stev = len(example_1[((mean_ex_1-2*stddev_ex_1)<example_1.iloc[:,-1]) & (example_1.iloc[:,-1]<(mean_ex_1+2*stddev_ex_1))])
ex_2_num_in_region_1_stev = len(example_2[((mean_ex_2-stddev_ex_2)<example_2.iloc[:,-1]) & (example_2.iloc[:,-1] <(mean_ex_2+stddev_ex_2))])
ex_2_num_in_region_2_stev = len(example_2[((mean_ex_2-2*stddev_ex_2)<example_2.iloc[:,-1]) & (example_2.iloc[:,-1]<(mean_ex_2+2*stddev_ex_2))])

print("*"*100)
print("Example 1: Percentage of observations in a region of a standard deviation from the mean:",
      round(100*ex_1_num_in_region_1_stev/len(example_1),2),"%.")
print("Example 1: Percentage of observations in a region of a 2x standard deviation from the mean:",
      round(100*ex_1_num_in_region_2_stev/len(example_1),2),"%.")
print("*"*100)
print("Example 2: Percentage of observations in a region of a standard deviation from the mean:",
      round(100*ex_2_num_in_region_1_stev/len(example_2),2),"%.")
print("Example 2: Percentage of observations in a region of a 2x standard deviation from the mean:",
      round(100*ex_1_num_in_region_2_stev/len(example_2),2),"%.")
print("*"*100)


########################################################################################################################
# Save the results from the training set for further analysis.
########################################################################################################################

df_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Historic_mean_and_stddevs_train.csv")

########################################################################################################################
# Compare the variability of the test set and compare it to the variability of the training set.
########################################################################################################################

# Use the test set data
modified_timeseries_test = modified_timeseries.iloc[62476:62476+7810]

# Compute the mean and variation for each x.
df_stats_test = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    df_stats_test.iloc[i-1,1]=np.mean(modified_timeseries_test[modified_timeseries_test["SP"]==i].iloc[:,-1])
    df_stats_test.iloc[i-1,2]=np.std(modified_timeseries_test[modified_timeseries_test["SP"]==i].iloc[:,-1])

zeros = np.zeros((336,))

# Plot the mean and variation for each x.
fig9, axs9=plt.subplots(1,1,figsize=(12,6))
axs9.fill_between(df_stats.iloc[:,0],
                  zeros,
                  +df_stats_test.iloc[:,2],
                  alpha=0.2, color = "black", label = "Standard deviation in the test set")
axs9.fill_between(df_stats.iloc[:,0],
                  zeros,
                  df_stats.iloc[:,2],
                  alpha=0.2, color = "blue", label = "Standard deviation in the training set")
axs9.set_ylabel("Electricity Load Training and Test Set, GW", size = 14)
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
plt.xticks(np.arange(1,385, 24), ["00:00 \nMonday", "12:00",
                                  "00:00 \nTuesday", "12:00",
                                  "00:00 \nWednesday", "12:00",
                                  "00:00 \nThursday", "12:00",
                                  "00:00 \nFriday", "12:00",
                                  "00:00 \nSaturday", "12:00",
                                  "00:00 \nSunday","12:00",
                                  "00:00"])
axs9.legend(fontsize = 12)
axs9.minorticks_on()
axs9.grid(b=True, which='major'), axs9.grid(b=True, which='minor',alpha = 0.2)
axs9.tick_params(axis = "both", labelsize = 12)
axs9.grid(True)
axs9.set_axisbelow(True)
fig9.show()
fig9.savefig("Load_Prediction/Historic_Load/Figures/Mean_and_Stddev_Train_vs_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Save the results from the test set for further analysis.
########################################################################################################################

df_stats_test.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Mean_and_stddevs_test.csv")

########################################################################################################################
# Plot the standard deviation of the test set.
########################################################################################################################

# Compare the mean and standard deviations of errors of the SVR between predictions and true values of the training set.
fig10, axes10 = plt.subplots(1,1,figsize=(12,6))
axes10.fill_between(df_stats_test.iloc[:,0],
                    zeros,
                    df_stats_test.iloc[:,-1],
                    label= "Standard deviation in the test set", alpha=0.2, color = "black")
axes10.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes10.set_xticks(np.arange(1,385, 24))
axes10.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes10.grid(True)
axes10.minorticks_on()
axes10.grid(b=True, which='major')
axes10.grid(b=True, which='minor',alpha = 0.2)
axes10.legend(fontsize=14)
axes10.tick_params(axis = "both", labelsize = 11)
axes10.set_ylim([0,3.65])
axes10.set_axisbelow(True)
fig10.show()
fig10.savefig("Load_Prediction/Historic_Load/Figures/Variability_Test.pdf", bbox_inches='tight')


