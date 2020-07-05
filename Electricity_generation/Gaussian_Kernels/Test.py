import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, ExpSineSquared as Exp, DotProduct as Lin
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
import datetime

np.random.seed(1)

X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X=X[:450,:]
y=y[:450,:]

X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)


y_train = np.asarray([y_tr[:,0], X_train[:,0], X_train[:,1], X_train[:,2], X_train[:,3], X_train[:,4]]).T

# y_scaler = StandardScaler()
# y_train = y_scaler.fit_transform(y_train)

# kernel_1 = Exp(length_scale=1, periodicity=1)
# #Weekly and daily variations could be modelled by periodic kernels
# #Multiplying the periodic kernel with the squared exponential kernel gives the periodic kernel some form of locality.
# kernel_2 = (Exp(length_scale=48, periodicity=48*7, length_scale_bounds=(1e-05,2)) + Exp(length_scale=48, periodicity=48, length_scale_bounds=(1e-05,2))) * RBF(length_scale=48)
# #The small variations could be modelled with a squared exponential kernel with short length scale.
# kernel_3 = Exp(length_scale=1, periodicity=48, length_scale_bounds=(1e-05,2))
# #Construct your own kernel.
# kernel = kernel_1 + kernel_2 + kernel_3
#*Exp(length_scale=48, periodicity=48)

#kernel = RBF(length_scale=48)
kernel =  RBF(length_scale=48)*Exp(length_scale=48, periodicity=48*7)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)  #Choose your amount of optimizer restarts.
x_train = np.atleast_2d(np.linspace(1, 360, 360)).T
gp.fit(x_train, y_train)

x_test = np.atleast_2d(np.linspace(360, 450, 90)).T
y_pred_1, sigma_1 = gp.predict(x_test, return_std=True)

# y_train = y_scaler.inverse_transform(y_train)
# y_pred_1 = y_scaler.inverse_transform(y_pred_1)
# #y_pred_train = y_scaler.inverse_transform(y_pred_train)
# sigma_1 = y_scaler.inverse_transform(sigma_1).reshape(-1,1)

plt.plot(x_train, y_tr[:,0], label=u'Training')
plt.plot(x_test, y_te[:,0], label=u'Observation')
plt.plot(x_test, y_pred_1[:,0], linewidth=1, label=u'Prediction')
lower = (y_pred_1 - sigma_1).reshape(-1,1)
upper = (y_pred_1 + sigma_1).reshape(-1,1)
plt.fill_between(x_test[:,0],lower[:,0], upper[:,0], alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('Prediction on the test set.')
plt.legend(loc='upper right', fontsize=10)
