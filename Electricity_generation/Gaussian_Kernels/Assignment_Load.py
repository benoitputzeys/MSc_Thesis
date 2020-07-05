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

from numpy import genfromtxt
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')

X = X[:200,:]
y = X[:,0]

#---------------------------------------------------------------------------------------------------

X_values = np.atleast_2d(np.linspace(1, 150, 200)).T  #Fill in the amount of test points you want.
x_values = np.atleast_2d(np.linspace(1, 200, 300)).T  #Fill in the amount of test points you want.

#---------------------------------------------------------------------------------------------------

kernel = C()*Exp(length_scale=2,periodicity=1)  #Construct your own kernel. Example: C()*Exp(length_scale=24,periodicity=1).
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=4)  #Choose your amount of optimizer restarts.

gp.fit(X_values, y)
y_pred_1, sigma_1 = gp.predict(x_values, return_std=True)

#---------------------------------------------------------------------------------------------------

kernel = C()*RBF(length_scale=20, length_scale_bounds=(1e-5, 2))  #Construct your own kernel. Example: C()*RBF(length_scale=24, length_scale_bounds=(1e-5, 2))
gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=4)  #Choose your amount of optimizer restarts.

gp.fit(X_values, y)
y_pred_2, sigma_2 = gp.predict(x_values, return_std=True)

#---------------------------------------------------------------------------------------------------

plt.subplot(2, 1, 1)
plt.plot(X_values, y.reshape(-1,1)[:, 0], 'r.', markersize=5, label=u'Observation')
plt.plot(x_values, y_pred_1.reshape(-1,1)[:, 0], 'b-', linewidth=1, label=u'Prediction')
plt.fill_between(x_values[:, 0], y_pred_1.reshape(-1,1)[:, 0] - 1.96 * sigma_1, y_pred_1.reshape(-1,1)[:, 0] + 1.96 * sigma_1, alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('(a)')
plt.legend(loc='upper right', fontsize=10)

plt.subplot(2, 1, 2)
plt.plot(X_values, y.reshape(-1,1)[:, 0], 'r.', markersize=5, label=u'Observation')
plt.plot(x_values, y_pred_2.reshape(-1,1)[:, 0], 'b-', linewidth=1, label=u'Prediction')
plt.fill_between( x_values[:, 0], y_pred_2.reshape(-1,1)[:, 0] - 1.96 * sigma_2, y_pred_2.reshape(-1,1)[:, 0] + 1.96 * sigma_2, alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('(b)')
plt.legend(loc='upper right', fontsize=10)
