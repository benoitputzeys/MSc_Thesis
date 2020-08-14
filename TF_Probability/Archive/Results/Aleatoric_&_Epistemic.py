import pandas as pd
import numpy as np
import matplotlib.ticker as plticker
from matplotlib import pylab as plt

df_NN_prediction = pd.read_csv('TF_Probability/Results/NN_prediction.csv')
df_projected_data = pd.read_csv('TF_Probability/Results/Projected_Data')

y_test = df_NN_prediction.iloc[:,-1]

var_NN = df_NN_prediction.iloc[:,1]**2
var_projected = df_projected_data.iloc[:,-1]**2

new_var = var_NN+var_projected
NN_mean = df_NN_prediction.iloc[:,0]

x_axis = np.linspace(1,336,336)
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].plot(x_axis,y_test, label = "Test Set", alpha = 1, color = "red")
axs2[0].plot(x_axis,NN_mean,label = "Mean of the NN predictions", color = "blue")
# Potentially include all the predictions made
#fig2.plot(X_axis[-48*7:], predictions.T[:,:50], alpha = 0.1, color = "blue")
axs2[0].fill_between(x_axis, NN_mean+np.sqrt(new_var), NN_mean-np.sqrt(new_var), alpha=0.3, color = "blue", label = "+- stddev")
axs2[0].set_ylabel('Load [MW]')
axs2[0].grid(True)

axs2[1].plot(x_axis,np.sqrt(new_var), label = "Test Set", alpha = 1, color = "blue")
axs2[1].set_xlabel('Settlement Periods (Test Set)')
axs2[1].set_ylabel('Standard Deviation')
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate()
axs2[1].grid(True)
fig2.legend()
fig2.show()