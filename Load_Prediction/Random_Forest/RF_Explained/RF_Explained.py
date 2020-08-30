# Import the necessary modules and libraries
# Code mostly taken from: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn import tree

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit 2 regression models with the same depth but different number of trees.
regr_1 = RandomForestRegressor(n_estimators=5, random_state=0, max_depth = 4)
regr_2 = RandomForestRegressor(n_estimators=100, random_state=0, max_depth = 4)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Make predictions with the 2 models.
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results with predictions in orange and green. The data is in blue.
fig1, axs1=plt.subplots(1,1,figsize=(8,6))
axs1.scatter(X, y, s=20, color="blue", label="Datapoint")
axs1.plot(X_test, y_1, color="orange",label="RF Prediction\nNumber of Trees = 5", linewidth=2)
axs1.plot(X_test, y_2, color="yellowgreen", label="RF Prediction\nNumber of Trees = 100", linewidth=2)
axs1.set_xlabel("x", size = 14)
axs1.set_ylabel("f(x)", size = 14)
axs1.set_title("Random Forest Regression", size = 14)
axs1.grid(True)
axs1.tick_params(axis = "both", labelsize = 14)
axs1.legend()
fig1.show()
# Save the figure.
fig1.savefig("Load_Prediction/Random_Forest/Figures/Random_Forest_Explained.pdf", bbox_inches='tight')
