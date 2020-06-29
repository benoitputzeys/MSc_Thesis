from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas import DataFrame
import datetime

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(features_df[i, -1]),
                                   month=int(features_df[i, -2]),
                                   day=int(features_df[i, -3]),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=(i % 2) * 30) for i in range(len(features_df))]
    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates


# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

mean = np.mean(y_train)
mean_zoom = np.mean(y_train[-48*3:])

print("-"*200)
sum_deviation_squared = sum((y_train-mean)**2)
variance = sum_deviation_squared/len(y_train)
print("The variance of the test set is %0.2f" % variance)
print("The variance using np array %0.2f" % np.var(y_train))
print("The standard deviation of the test set is %0.2f [MW]" % np.sqrt(variance))
print("The standard deviation using np array %0.2f [MW]" % np.std(y_train))
print("-"*200)

fig, axes = plt.subplots(3)
fig.subtitle = ("Actual Values and Volatility")
y_values = create_dates(X_train,y_train)
axes[0].plot(y_values,label = "Actual historic values")
y_values = create_dates(X_train,mean)
axes[0].plot(y_values, label = "Mean of historic values",color = 'orange')
axes[0].set_xlabel('Settlement Period')
axes[0].set_ylabel('Electricity Load [MW]')

y_values = create_dates(X_train,y_train-mean)
axes[1].plot(y_values,label = "Difference between mean and actual load",color = 'black')
axes[1].set_xlabel('Settlement Period')
axes[1].set_ylabel('Difference')

y_values = create_dates(X_train,y_train-mean)
axes[2].plot(y_values**2,label = "Squarred Difference",color = 'red')
axes[2].set_xlabel('Settlement Period')
axes[2].set_ylabel('Squared Difference')
fig.legend()
fig.show()

dates = [datetime.datetime(year=int(X_train[-i, -1]),
                   month=int(X_train[-i, -2]),
                   day=int(X_train[-i, -3]),
                   hour=int((X_train[-i, -4] - 1) / 2),
                   minute=(-i % 2) * 30) for i in range(48*3,0,-1)]

fig1, axes1 = plt.subplots(3)
fig1.subtitle = ("Actual Values and Volatility")
y_values = create_dates(X_train[-48*3:],y_train[-48*3:])
axes1[0].plot(y_values,label = "Actual historic values")
axes1[0].fill_between(dates,  y_train[-48*3:][:,0]-np.std(y_train),  y_train[-48*3:][:,0]+np.std(y_train),alpha=0.1)
y_values = create_dates(X_train[-48*3:],mean_zoom)
axes1[0].set_xlabel('Settlement Period')
axes1[0].set_ylabel('Electricity Load [MW]')

y_values = create_dates(X_train[-48*3:],y_train[-48*3:]-mean)
axes1[1].plot(y_values,label = "Difference between mean and actual load",color = 'black')
axes1[1].set_xlabel('Settlement Period')
axes1[1].set_ylabel('Difference')

y_values = create_dates(X_train[-48*3:],y_train[-48*3:]-mean)
axes1[2].plot(y_values**2,label = "Squarred Difference",color = 'red')
axes1[2].set_xlabel('Settlement Period')
axes1[2].set_ylabel('Squared Difference')
fig1.legend()
fig1.show()

