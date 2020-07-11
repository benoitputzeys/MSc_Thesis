import matplotlib as mpl
from matplotlib import pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import collections
import tensorflow.compat.v2 as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_generation.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from numpy import genfromtxt
import matplotlib.ticker as plticker
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import datetime

tf.enable_v2_behavior()

def plot_forecast(x, y,
                  forecast_mean, forecast_scale, forecast_samples,
                  title, x_locator=None, x_formatter=None):
  """Plot a forecast distribution against the 'true' time series."""
  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]
  fig = plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1, 1, 1)

  num_steps = len(y)
  num_steps_forecast = forecast_mean.shape[-1]
  num_steps_train = num_steps - num_steps_forecast


  ax.plot(x[-48*3:], y[-48*3:], lw=2, color="red", label='ground truth')

  forecast_steps = np.arange(
      x[num_steps_train],
      x[num_steps_train]+num_steps_forecast,
      dtype=x.dtype)

  ax.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)

  ax.plot(forecast_steps, forecast_mean, lw=2, color="blue",
           label='forecast')
  ax.fill_between(forecast_steps,
                   forecast_mean-2*forecast_scale,
                   forecast_mean+2*forecast_scale, color="blue", alpha=0.2)

  ymin, ymax = min(np.min(forecast_samples), np.min(y)), max(np.max(forecast_samples), np.max(y))
  yrange = ymax-ymin
  ax.set_ylim([ymin - yrange*0.1, ymax + yrange*0.1])
  ax.set_title("{}".format(title))
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()

  return fig, ax

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

def plot_one_step_predictive(dates, observed_time_series,
                             one_step_mean, one_step_scale,
                             x_locator=None, x_formatter=None):
  """Plot a time series against a model's one-step predictions."""

  colors = sns.color_palette()
  c1, c2 = colors[0], colors[1]

  fig=plt.figure(figsize=(12, 6))
  ax = fig.add_subplot(1,1,1)
  num_timesteps = one_step_mean.shape[-1]
  ax.plot(dates, observed_time_series, label="observed time series", color=c1)
  ax.plot(dates, one_step_mean, label="one-step prediction", color=c2)
  ax.fill_between(dates,
                  one_step_mean - one_step_scale,
                  one_step_mean + one_step_scale,
                  alpha=0.1, color=c2)
  ax.legend()

  if x_locator is not None:
    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)
    fig.autofmt_xdate()
  fig.tight_layout()
  return fig, ax

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/X_API.csv')
X = X.iloc[1:,1:]
# Extract the dates and delete the respective column.
dates = X.iloc[:,-1]
X = X.iloc[:,:-1]

y = pd.read_csv('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/y_API.csv')
y = y.iloc[1:,2:]

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# Plot the training set.
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(dates[:len(y_train)], y_train/1000, label="training data")
axs1.set_ylabel("Load [GW]")
axs1.set_xlabel("Settlement Periods")
#fig1.suptitle("Testing TF Proability",fontsize=15, x = 0.5, y = 0.99)
loc = plticker.MultipleLocator(base=4800) # this locator puts ticks at regular intervals
axs1.xaxis.set_major_locator(loc)
axs1.grid(True)
fig1.autofmt_xdate()
plt.show()

# Specify how many steps you want to forecast. In 1 day there are 48 settlement periods.
# To predict 1 week in advance, set this to 48*7.
num_forecast_steps = 48 * 7

def build_model(observed_time_series):
  seasonal_day = tfp.sts.Seasonal(
      num_seasons=48,
      observed_time_series=observed_time_series,
      name = 'Daily_Seasonality')
  seasonal_week = tfp.sts.Seasonal(
      num_seasons=7,
      num_steps_per_season=48,
      observed_time_series=observed_time_series,
      name = 'Weekly_Seasonality')
  seasonal_year = tfp.sts.Seasonal(
      num_seasons=1,
      num_steps_per_season=48*365,
      observed_time_series=observed_time_series,
      name = 'Yearly_Seasonality')
  trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
  autoregressive = sts.Autoregressive(
      order=1,
      observed_time_series=observed_time_series,
      name='autoregressive')
  model = sts.Sum([seasonal_day,
                   seasonal_week,
                   seasonal_year,
                   trend,
                   autoregressive],
                  observed_time_series=observed_time_series)
  return model

load_model = build_model(y_train)

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

elbo_loss_curve = train()

plt.plot(elbo_loss_curve)
plt.show()

# Draw samples from the variational posterior.
q_samples_load_ = variational_posteriors.sample(50)

print("Inferred parameters:")
for param in load_model.parameters:
  print("{}: {} +- {}".format(param.name,
                              np.mean(q_samples_load_[param.name], axis=0),
                              np.std(q_samples_load_[param.name], axis=0)))

load_forecast_dist = tfp.sts.forecast(
      load_model,
      observed_time_series=y_train,
      parameter_samples=q_samples_load_,
      num_steps_forecast=num_forecast_steps)

num_samples=5

load_forecast_mean, load_forecast_scale, load_forecast_samples = (
    load_forecast_dist.mean().numpy()[..., 0],
    load_forecast_dist.stddev().numpy()[..., 0],
    load_forecast_dist.sample(num_samples).numpy()[..., 0])

fig2, ax2 = plot_forecast(dates[:len(y_train)], y_train,
                        load_forecast_mean, load_forecast_scale, load_forecast_samples,
                        title="Load forecast")

fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates[len(y_train)-48*3:len(y_train)], y_train[-48 * 3:], color="red", label='Training Set')
axs2.plot(dates[len(y_train):48*7], load_forecast_mean, color="blue",label='Forecast with 2x standard deviation')
axs2.fill_between(dates[len(y_train):len(y_train)+48*7],load_forecast_mean-2*load_forecast_scale,load_forecast_mean+2*load_forecast_scale, color="blue", alpha=0.2)
axs2.axvline(dates[len(y_train):48*7], linestyle="--", color = "black")
axs2.title("SARIMA model with probabilistic prediction")
axs2.set_xlabel("Settelement Periods")
axs2.set_ylabel("Load [MW]")
axs2.legend(loc = "lower left")
loc = plticker.MultipleLocator(base=4800) # this locator puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc)
axs2.grid(True)
fig2.autofmt_xdate()
axs2.show()
plt.show()

# Build a dict mapping components to distributions over
# their contribution to the observed signal.
component_dists = sts.decompose_by_component(
    load_model,
    observed_time_series=y_train,
    parameter_samples=q_samples_load_)

load_component_means_, load_component_stddevs_ = (
    {k.name: c.mean() for k, c in component_dists.items()},
    {k.name: c.stddev() for k, c in component_dists.items()})

_ = plot_components(dates, load_component_means_, load_component_stddevs_,
                    x_locator=None, x_formatter=None)
plt.show()


# Calculate the errors
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
with open('TF_Probability/Results/SARIMA_result.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["SARIMA",
                     str(np.mean(errors**2)),
                     str(np.mean(errors)),
                     str(np.sqrt(np.mean(errors**2)))
                     ])
