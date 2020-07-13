from matplotlib import pylab as plt
import seaborn as sns
import collections
import tensorflow.compat.v2 as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability import sts
tf.enable_v2_behavior()
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X[1:48*50,1:-1]

y = genfromtxt('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y[1:48*50,-1]

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

num_forecast_steps = 48 * 7

X_axis = np.linspace(1,len(X_train),len(X_train))

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_axis, y_train, label="training data", color = "blue", linewidth = 0.5)
ax.set_ylabel("Load [MW]")
ax.set_xlabel("Settlement Periods")
plt.show()

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
seasonal_annual = tfp.sts.Seasonal(
    num_seasons=1,
    num_steps_per_season=48*365,
    observed_time_series=y_train,
    name = 'Annual_Seasonality')
load_model = sts.Sum([seasonal_day,
                 seasonal_week,
                 seasonal_annual],
                observed_time_series=y_train)

plt.plot(seasonal_day)
plt.show()

# Build the variational surrogate posteriors `qs`.
variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=load_model)

# Allow external control of optimization to reduce test runtimes.
num_variational_steps = 100 # @param { isTemplate: true}
num_variational_steps = int(num_variational_steps)

optimizer = tf.optimizers.Adam(learning_rate=.1)

elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=load_model.joint_log_prob(observed_time_series=y_train),
    surrogate_posterior=variational_posteriors,
    optimizer=optimizer,
    num_steps=num_variational_steps)

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

num_samples=10

load_forecast_mean, load_forecast_scale, load_forecast_samples = (
    load_forecast_dist.mean().numpy()[..., 0],
    load_forecast_dist.stddev().numpy()[..., 0],
    load_forecast_dist.sample(num_samples).numpy()[..., 0])

dates = np.linspace(1,len(X),len(X))

fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates[len(y_train)-48*3:len(y_train)], y_train[-48 * 3:]/1000, color="black", label='Training Set')
axs2.plot(dates[len(y_train):len(y_train)+48*7], y_test[:48 * 7]/1000, color="red", label = "Test Set")
axs2.plot(dates[len(y_train):48*7+len(y_train)], load_forecast_mean/1000, color="blue",label='Forecast with 2x standard deviation')
axs2.fill_between(dates[len(y_train):len(y_train)+48*7],(load_forecast_mean-2*load_forecast_scale)/1000,(load_forecast_mean+2*load_forecast_scale)/1000, color="blue", alpha=0.2)
axs2.axvline(dates[len(y_train)], linestyle="--", color = "black")
axs2.set_xlabel("Settelement Periods")
axs2.set_ylabel("Load [GW]")
axs2.legend(loc = "lower left")
#loc = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
#axs2.xaxis.set_major_locator(loc)
axs2.grid(True)
#fig2.autofmt_xdate()
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
