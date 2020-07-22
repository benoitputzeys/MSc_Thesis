from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.ticker as plticker

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# Naive prediction
pred = X_test.iloc[:48*7,0]

error = np.zeros((336+48*3,1))
error[-336:,0] = np.abs(X_test.iloc[:48*7,0]-y_test.iloc[:48*7,0])

# Plot the result with the truth in red and the predictions in blue.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].grid(True)
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],y_train.iloc[-48*3:,0]/1000, label = "Training Set (True Values)", alpha = 1, color = "black")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7], pred/1000, label = "Naive Prediction", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],y_test.iloc[:48*7,0]/1000, label = "Test Set (True Values)", alpha = 1, color = "blue")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)

axs2[1].grid(True)
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],error/1000, label = "Absolute Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Absolute Error [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=15)

axs2[1].legend(loc=(1.04,0.9))
axs2[0].legend(loc=(1.04,0.6))

fig2.show()

# Calculate the errors from the mean to the actual vaules.
error_test = np.abs(X_test.iloc[:,0]-y_test.iloc[:,0])

print("-"*200)
print("The mean absolute error of the entire test set is %0.2f" % np.mean(error_test))
print("The mean squared error of the entire test set is %0.2f" % np.mean(error_test**2))
print("The root mean squared error of the entire test set is %0.2f" % np.sqrt(np.mean(error_test**2)))
print("-"*200)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Compare_Models/Single_Multi_Step_results/Naive.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["Naive",
                     str(np.mean(error_test**2)),
                     str(np.mean(error_test)),
                     str(np.sqrt(np.mean(error_test**2)))
                     ])