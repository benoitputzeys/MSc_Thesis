from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas import DataFrame
import datetime

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(round(features_df[i, -1])),
                                   month=int(round(features_df[i, -2])),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=int(((features_df[i, -4] -1) % 2 ) * 30)) for i in range(len(features_df))]

    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

########################################################################################################################
# Print the training set, the mean values, the difference and the squared difference between actual and mean values.
########################################################################################################################

mean = np.mean(y_train)
mean_zoom = np.mean(y_train[-48*3:])

print("-"*200)
print("The variance of the training set is %0.2f" % np.var(y_train))
print("The standard deviation of the training set is %0.2f [MW]" % np.std(y_train))
print("-"*200)

fig, axes = plt.subplots(3)

left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.25  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.35  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

fig.subtitle = ("Actual Values and Volatility")
axes[0].plot(create_dates(X_train,y_train),label = "Actual historic values", linewidth=0.5)
axes[0].plot(create_dates(X_train,mean), label = "Mean of historic values",color = 'orange', linewidth=0.5)
axes[0].set_xlabel('Settlement Period Training Set')
axes[0].set_ylabel('Electricity Load [MW]')

axes[1].plot(create_dates(X_train,y_train-mean),label = "Difference between mean and actual load",color = 'black', linewidth=0.5)
axes[1].set_xlabel('Settlement Period Training Set')
axes[1].set_ylabel('Difference (Mean-Actual) [MW]')

axes[2].plot(create_dates(X_train,y_train-mean)**2,label = "Squarred Difference",color = 'red', linewidth=0.5)
axes[2].set_xlabel('Settlement Period Training Set')
axes[2].set_ylabel('Squared Difference')
fig.legend()
fig.show()

########################################################################################################################
# Print the volatility of the last 7 days of the training set.
########################################################################################################################

dates = [datetime.datetime(year=int(X_train[-i, -1]),
                   month=int(X_train[-i, -2]),
                   day=int(X_train[-i, -3]),
                   hour=int((X_train[-i, -4] - 1) / 2),
                   minute=int(((X_train[i, -4] -1) % 2 ) * 30)) for i in range(48*3,0,-1)]

fig1, axes1 = plt.subplots(3)

left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.25  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.35  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

fig1.subtitle = ("Actual Values and Volatility")
axes1[0].plot(create_dates(X_train[-48*3:],y_train[-48*3:]), label = "Actual historic values", linewidth=0.5)
axes1[0].fill_between(dates,  y_train[-48*3:][:,0] - np.std(y_train), y_train[-48*3:][:,0] + np.std(y_train), alpha=0.1)
axes1[0].plot(create_dates(X_train[-48*3:],mean_zoom), label = "Mean train set", linewidth=0.5)
axes1[0].set_xlabel('Settlement Period')
axes1[0].set_ylabel('Electricity Load [MW]')

axes1[1].plot(create_dates(X_train[-48*3:],y_train[-48*3:] - mean), label = "Difference between mean and actual load",color = 'black', linewidth=0.5)
axes1[1].set_xlabel('Settlement Period')
axes1[1].set_ylabel('Difference (Mean-Actual) [MW]')

axes1[2].plot(create_dates(X_train[-48*3:],y_train[-48*3:]-mean)**2,label = "Squared Difference",color = 'red', linewidth=0.5)
axes1[2].set_xlabel('Settlement Period')
axes1[2].set_ylabel('Squared Difference')
fig1.legend()
fig1.show()

########################################################################################################################
# Print the volatility of the training set.
########################################################################################################################

dates = [datetime.datetime(year=int(X_train[-i, -1]),
                   month=int(X_train[-i, -2]),
                   day=int(X_train[-i, -3]),
                   hour=int((X_train[-i, -4] - 1) / 2),
                   minute=int(((X_train[i, -4] -1) % 2 ) * 30)) for i in range(len(X_train)-1,0,-1)]

fig1, axes1 = plt.subplots(3)
fig1.subtitle = ("Actual Values and Volatility")
y_values = create_dates(X_train,y_train)
axes1[0].plot(y_values,label = "Actual historic values", linewidth=0.5)
axes1[0].fill_between(dates,  y_train[:,0]-np.std(y_train),  y_train[:,0]+np.std(y_train),alpha=0.1)
y_values = create_dates(X_train,mean_zoom)
axes1[0].set_xlabel('Settlement Period')
axes1[0].set_ylabel('Electricity Load [MW]')

axes1[1].plot(create_dates(X_train,y_train-mean),label = "Difference between mean and actual load",color = 'black', linewidth=0.5)
axes1[1].set_xlabel('Settlement Period')
axes1[1].set_ylabel('Difference')

axes1[2].plot(create_dates(X_train,y_train-mean)**2,label = "Squared Difference",color = 'red', linewidth=0.5)
axes1[2].set_xlabel('Settlement Period')
axes1[2].set_ylabel('Squared Difference')
fig1.legend()
fig1.show()

