#This code forms part of the course Gaussian Process Regression for Bayesian Machine Learning, wich is available at https://www.udemy.com/course/gaussian-process-regression-fundamentals-and-application/?src=sac&kw=gaussian+process+regre

#Follow the course for an explanation of the code as well as a fundamental understanding of Gaussian process regression.

#---------------------------------
#Do not edit from here

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, ExpSineSquared as Exp, DotProduct as Lin

np.random.seed(1)

df = pd.read_csv('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Gaussian_Kernels/weather_data.csv', sep=';')

df_array = np.asarray(df)

date = df_array[0:177, 0]
rec_num = df_array[0:177, 1]
ghi = df_array[0:177, 2]
DNI = df_array[0:177, 3]
DHI = df_array[0:177, 4]
DHI_shadowband = df_array[0:177, 5]
UVA = df_array[0:177, 6]
UVB = df_array[0:177, 7]
air_temp = df_array[0:177, 8]
BP = df_array[0:177, 9]
RH = df_array[0:177, 10]
WS = df_array[0:177, 11]
WD = df_array[0:177, 12]
WD_SD = df_array[0:177, 13]

y = np.asarray(
    [ghi, DNI, DHI, DHI_shadowband, UVA, UVB, air_temp, BP, RH, WS, WD,
     WD_SD]).T

#---------------------------------------------------------------------------------------------------

X = np.atleast_2d(np.linspace(1, 177, 177)).T  #Fill in the amount of test points you want.
x = np.atleast_2d(np.linspace(1, 200, 1000)).T  #Fill in the amount of test points you want.

#---------------------------------------------------------------------------------------------------


kernel = C()*Exp(length_scale=24,periodicity=1)  #Construct your own kernel. Example: C()*Exp(length_scale=24,periodicity=1).

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=4)  #Choose your amount of optimizer restarts.

gp.fit(X, y)
y_pred_1, sigma_1 = gp.predict(x, return_std=True)

#---------------------------------------------------------------------------------------------------

kernel = C()*RBF(length_scale=24, length_scale_bounds=(1e-5, 2))  #Construct your own kernel. Example: C()*RBF(length_scale=24, length_scale_bounds=(1e-5, 2))

gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=4)  #Choose your amount of optimizer restarts.

gp.fit(X, y)
y_pred_2, sigma_2 = gp.predict(x, return_std=True)

#---------------------------------------------------------------------------------------------------

plt.subplot(2, 1, 1)
plt.plot(X, y[:, 0], 'r.', markersize=5, label=u'Observation')
plt.plot(x, y_pred_1[:, 0], 'b-', linewidth=1, label=u'Prediction')
plt.fill_between(x[:, 0], y_pred_1[:, 0] - 1.96 * sigma_1, y_pred_1[:, 0] + 1.96 * sigma_1, alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('(a)')
plt.legend(loc='upper right', fontsize=10)

plt.subplot(2, 1, 2)
plt.plot(X, y[:, 0], 'r.', markersize=5, label=u'Observation')
plt.plot(x, y_pred_2[:, 0], 'b-', linewidth=1, label=u'Prediction')
plt.fill_between( x[:, 0], y_pred_2[:, 0] - 1.96 * sigma_2, y_pred_2[:, 0] + 1.96 * sigma_2, alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('(b)')
plt.legend(loc='upper right', fontsize=10)
