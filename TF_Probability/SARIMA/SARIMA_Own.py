from matplotlib import pylab as plt
import seaborn as sns
import collections
import tensorflow.compat.v2 as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import sts
tf.enable_v2_behavior()
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import pandas as pd
import matplotlib.ticker as plticker

def plot_components(dates,
                    component_means_dict,
                    component_stddevs_dict,
                    x_locator=None,
                    x_formatter=None):
  """Plot the contributions of posterior components in a single figure."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  axes_dict = collections.OrderedDict()
  num_components = len(component_means_dict)
  fig = plt.figure(figsize=(12, 2.5 * num_components))
  for i, component_name in enumerate(component_means_dict.keys()):
    component_mean = component_means_dict[component_name]
    component_stddev = component_stddevs_dict[component_name]

    ax = fig.add_subplot(num_components,1,1+i)
    ax.plot(dates, component_mean, lw=2)
    ax.fill_between(dates,
                     component_mean-2*component_stddev,
                     component_mean+2*component_stddev,
                     color=c2, alpha=0.5)
    ax.set_title(component_name)
    if x_locator is not None:
      ax.xaxis.set_major_locator(x_locator)
      ax.xaxis.set_major_formatter(x_formatter)
    axes_dict[component_name] = ax
  fig.autofmt_xdate()
  fig.tight_layout()
  return fig, axes_dict

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
DoW = X["Day of Week"]
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]
dates_train = dates[:len(X_train)]
dates_test = dates[-len(X_test):]

########################################################################################################################
# Build the model.
########################################################################################################################

#Decompose the data into daily and seasonal components
#Does not really have an upwards trend if only 2 weeks are considered as training.
#trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
seasonal_day = tfp.sts.Seasonal(
    num_seasons=48,
    observed_time_series=y_train[-48*7*2:],
    name = 'seasonal_day')
seasonal_week = tfp.sts.Seasonal(
    num_seasons=7,
    num_steps_per_season=48,
    observed_time_series=y_train[-48*7*2:],
    name = 'seasonal_week')
load_model = sts.Sum([seasonal_day,
                 seasonal_week],
                observed_time_series=y_train[-48*7*2:])

# Build the variational surrogate posteriors `qs`.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=load_model)

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 200 # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=0.1)
# Using fit_surrogate_posterior to build and optimize the variational loss function.
@tf.function(experimental_compile=True)

def train():
  elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=load_model.joint_log_prob(observed_time_series=y_train[-48*7*2:]),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)
  return elbo_loss_curve

# Train the model using the ELBO method.
elbo_loss_curve = train()

# Plot how the training converges.
fig3, axs3=plt.subplots(1,1)
axs3.plot(elbo_loss_curve, color = "blue")
axs3.set_xlabel("Iterations")
axs3.set_ylabel("Loss")
axs3.grid(True)
fig3.show()
fig3.savefig("TF_Probability/SARIMA/Figures/Loss_Curve.pdf", bbox_inches='tight')

########################################################################################################################
# Make a prediction.
########################################################################################################################

# Draw samples from the variational posterior.
q_samples_load_ = variational_posteriors.sample(50)

#Forecast the test set in advance.
num_forecast_steps = len(y_test)

load_forecast_dist = tfp.sts.forecast(
      load_model,
      observed_time_series=y_train[-48*7*2:],
      parameter_samples=q_samples_load_,
      num_steps_forecast=num_forecast_steps)

load_forecast_mean = load_forecast_dist.mean().numpy().reshape(-1,)
load_forecast_scale =  load_forecast_dist.stddev().numpy().reshape(-1,)

error_test_plot = np.zeros((len(X_test),1))
error_test_plot = np.array((load_forecast_mean-y_test.iloc[:,0])/1000).reshape(-1,1)

# Plot the actual values, the forecast and the standard deviation.
fig2, axs2=plt.subplots(2,1,figsize=(12,10))

axs2[0].plot(dates_test,
          y_test/1000,
          color="black", label = "Test Set")
axs2[0].plot(dates_test,
          load_forecast_mean/1000,
          color="orange",label='SARIMA Forecast with \n+- 1 x Standard Deviation')
axs2[0].fill_between(dates_test,
                  (load_forecast_mean-load_forecast_scale)/1000,
                  (load_forecast_mean+load_forecast_scale)/1000, color="orange", alpha=0.2)
axs2[0].set_xlabel("Dates",size = 14)
axs2[0].set_ylabel("Load, GW",size = 14)

axs2[1].plot(dates_test,error_test_plot, color="red", label='Training Set')
axs2[1].set_xlabel("Dates",size = 14)
axs2[1].set_ylabel("Error, GW",size = 14)

#axs2[0].axvline(dates_train[-1], linestyle="--", color = "black")
#axs2[1].axvline(dates_train[-1], linestyle="--", color = "black")
axs2[0].legend()
axs2[1].legend()
loc = plticker.MultipleLocator(base=48*25) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)
axs2[1].xaxis.set_major_locator(loc)
axs2[0].grid(True)
axs2[1].grid(True)
fig2.autofmt_xdate(rotation = 12)
fig2.show()
fig2.savefig("TF_Probability/SARIMA/Figures/Pred_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Plot the error per settlement period.
########################################################################################################################

settlement_period_test = X["Settlement Period"][-len(X_test)*2:-len(X_test)].values+(48*DoW[-len(X_test)*2:-len(X_test)]).values
long_column = np.array([settlement_period_test]).reshape(-1,)
# Create a dataframe that contains the SPs (1-336) and the load values.
mean_errors = load_forecast_mean - y_test.values[:,0]
error_test = pd.DataFrame({'SP':long_column, 'Means': mean_errors/1000,'Stddev': load_forecast_scale/1000 })

# Compute the mean and variation for each x.
test_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    test_stats.iloc[i-1,1]=np.mean(error_test[error_test["SP"]==i].iloc[:,1])
    test_stats.iloc[i-1,2]=np.mean(error_test[error_test["SP"]==i].iloc[:,2])

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
# Plot the mean and standard deviation of the errors that are made on the test set.
axs5.plot(test_stats.iloc[:,0],
          test_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors")
axs5.fill_between(test_stats.iloc[:,0],
                  (test_stats.iloc[:,1]-test_stats.iloc[:,2]),
                  (test_stats.iloc[:,1]+test_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs5.set_ylabel("Error Test Set, GW", size = 14)
axs5.set_xticks(np.arange(1,385, 48))
axs5.set_xticklabels(["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs5.legend()
axs5.grid(True)
fig5.show()
fig5.savefig("TF_Probability/SARIMA/Figures/Projected_Error_Test.pdf", bbox_inches='tight')

# Calculate the errors from the mean to the actual vaules.
print("-"*200)
errors = abs(mean_errors)/1000
print("The mean absolute error of the test set is %0.2f GW" % np.mean(errors))
print("The mean squared error of the test set is %0.2f GW" % np.mean(errors**2))
print("The root mean squared error of the test set is %0.2f GW" % np.sqrt(np.mean(errors**2)))
print("-"*200)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

# import csv
# with open('Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/SARIMA_error.csv', 'w', newline='', ) as file:
#     writer = csv.writer(file)
#     writer.writerow(["Method","MSE","MAE","RMSE"])
#     writer.writerow(["SARIMA",
#                      str(np.mean(errors**2)),
#                      str(np.mean(errors)),
#                      str(np.sqrt(np.mean(errors**2)))
#                      ])
# test_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/SARIMA_mean_errors_stddevs.csv")
