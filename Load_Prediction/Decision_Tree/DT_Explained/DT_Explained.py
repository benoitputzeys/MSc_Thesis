# Import the necessary modules and libraries
# Code mostly taken from: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression models with different maximal depths.
regr_1 = DecisionTreeRegressor(max_depth=3)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict valuies with the models.
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results with predictions in orange or green.
fig1, axs1=plt.subplots(1,1,figsize=(8,6))
axs1.scatter(X, y, s=20, color="blue", label="Datapoint")
axs1.plot(X_test, y_1, color="orange",label="DT Prediction\nMax Depth = 3", linewidth=2)
axs1.plot(X_test, y_2, color="yellowgreen", label="DT Prediction\nMax Depth = 5", linewidth=2)
axs1.set_xlabel("x", size = 14)
axs1.set_ylabel("f(x)", size = 14)

# Include additional details.
axs1.set_title("Decision Tree Regression", size = 14)
axs1.grid(True)
axs1.tick_params(axis = "both", labelsize = 14)
axs1.legend()
fig1.show()
fig1.savefig("Load_Prediction/Decision_Tree/Figures/Decision_Tree_Explained.pdf", bbox_inches='tight')

########################################################################################################################
# Plot the decision tree itself
########################################################################################################################

import pydotplus
import collections

dot_data = tree.export_graphviz(regr_1,
                                out_file=None,
                                filled=True,
                                rounded=True,
                                special_characters=True,
                                feature_names='x')
graph = pydotplus.graph_from_dot_data(dot_data)

nodes = graph.get_node_list()
edges = graph.get_edge_list()

# Define the colors for the DT.
colors = ('deepskyblue', 'royalblue','white')
edges = collections.defaultdict(list)

# Set the colors for the leaves in the tree.
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
nodes[2].set_fillcolor(colors[-1])

# Save the file.
graph.write_pdf('Load_Prediction/Decision_Tree/Figures/Tree_Decisions.pdf')

