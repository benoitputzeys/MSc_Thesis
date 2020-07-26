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
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
DoW = X["Day of Week"]
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
dates = dates[-len(X_train)-len(X_test):]
dates_train = dates[:len(X_train)]
dates_test = dates[-len(X_test):]

#Plot the training set
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(dates_train,
          y_train,
          label="training data", color = "blue", linewidth = 0.5)
axs1.set_ylabel("Load [MW]")
axs1.set_xlabel("Settlement Periods")
loc = plticker.MultipleLocator(base=48*70) # this locator puts ticks at regular intervals
axs1.xaxis.set_major_locator(loc)
axs1.grid(True)
fig1.autofmt_xdate(rotation = 12)
fig1.show()

#Decompose the data into daily and seasonal components
#Does not really have an upwards trend.
#trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
seasonal_day = tfp.sts.Seasonal(
    num_seasons=48,
    observed_time_series=y_train,
    name = 'Daily_Seasonality')
seasonal_week = tfp.sts.Seasonal(
    num_seasons=7,
    num_steps_per_season=48,
    observed_time_series=y_train,
    name = 'Weekly_Seasonality')
load_model = sts.Sum([seasonal_day,
                 seasonal_week],
                observed_time_series=y_train)

# Build the variational surrogate posteriors `qs`.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=load_model)

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 100 # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=.1)
# Using fit_surrogate_posterior to build and optimize the variational loss function.
@tf.function(experimental_compile=True)

def train():
  elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=load_model.joint_log_prob(observed_time_series=y_train),
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

# Draw samples from the variational posterior.
q_samples_load_ = variational_posteriors.sample(100)

# Necessary ??
print("Inferred parameters:")
for param in load_model.parameters:
  print("{}: {} +- {}".format(param.name,
                              np.mean(q_samples_load_[param.name], axis=0),
                              np.std(q_samples_load_[param.name], axis=0)))

#Forecast 1 week in advance.
num_forecast_steps = 48 * 7

load_forecast_dist = tfp.sts.forecast(
      load_model,
      observed_time_series=y_train,
      parameter_samples=q_samples_load_,
      num_steps_forecast=num_forecast_steps)

load_forecast_mean = load_forecast_dist.mean().numpy().reshape(-1,)
load_forecast_scale =  load_forecast_dist.stddev().numpy().reshape(-1,)

# # Optional: One can draw individual samples from the prediction. On a sidenote, one can reconstruct the standard deviation
# # by taking a lot of samples and computing the mean and average of them.
# num_samples=10
# load_forecast_samples = load_forecast_dist.sample(num_samples).numpy()
# mean_recons = np.zeros((336,1))
# for i in range(336):
#     mean_recons[i,0] = np.mean(load_forecast_samples[:,i])

# Plot the actual values, the forecast and the standard deviation.
fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates_train[-48*3:],
          y_train[-48 * 3:]/1000,
          color="blue", label='Training Set')
axs2.plot(dates_test[:48*7],
          y_test/1000,
          color="red", label = "Test Set")
axs2.plot(dates_test[:48*7],
          load_forecast_mean/1000,
          color="blue",label='Forecast with 2x standard deviation')
axs2.fill_between(dates_test[:48*7],
                  (load_forecast_mean-load_forecast_scale)/1000,
                  (load_forecast_mean+load_forecast_scale)/1000, color="blue", alpha=0.2)
axs2.axvline(dates_test[0], linestyle="--", color = "black")
axs2.set_xlabel("Settelement Periods",size = 14)
axs2.set_ylabel("Load [GW]",size = 14)
axs2.legend(loc = "lower left")
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc)
axs2.grid(True)
fig2.autofmt_xdate(rotation = 12)
fig2.show()

# Necessary ??
# # Build a dict mapping components to distributions over
# # their contribution to the observed signal.
# component_dists = sts.decompose_by_component(
#     load_model,
#     observed_time_series=y_train,
#     parameter_samples=q_samples_load_)
#
# load_component_means_, load_component_stddevs_ = (
#     {k.name: c.mean() for k, c in component_dists.items()},
#     {k.name: c.stddev() for k, c in component_dists.items()})
#
# _ = plot_components(X_axis, load_component_means_, load_component_stddevs_,
#                     x_locator=None, x_formatter=None)
# plt.show()

# Calculate the errors from the mean to the actual vaules.
print("-"*200)
errors = abs(load_forecast_mean.reshape(-1,1)-y_train[:48*7])
print("The mean absolute error of the test set is %0.2f" % np.mean(errors))
print("The mean squared error of the test set is %0.2f" % np.mean(errors**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(errors**2)))
print("-"*200)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('TF_Probability/Results/SARIMA_error.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["SARIMA",
                     str(np.mean(errors**2)),
                     str(np.mean(errors)),
                     str(np.sqrt(np.mean(errors**2)))
                     ])
with open('TF_Probability/Results/SARIMA_prediction.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","Mean","Stddev"])
    writer.writerow(["SARIMA",
                     str(load_forecast_mean),
                     str(load_forecast_scale),
                     ])
