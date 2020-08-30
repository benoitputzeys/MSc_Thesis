import numpy as np
import matplotlib.pyplot as plt

# Create a dataset with some randomly distributed points.
x_axis = np.linspace(1,10,num = 100)
# Create a straight line
list1 = x_axis + np.random.normal(0,1,100)
# Define the margin
a = 3.05

# Plot the straingt line, the datapoints and the margin defined by a
fig1, axs1=plt.subplots(1,1,figsize=(8,6))
axs1.scatter(x_axis, list1, label = "Datapoint",color = "blue")
axs1.plot(x_axis, x_axis, label = "Prediction \nf(x)", color = "red")
axs1.plot(x_axis, x_axis+a, '--', label = "Margin \n(+- ε)", color = "orange")
axs1.plot(x_axis, x_axis-a, '--', color = "orange")
axs1.set_xlabel('x',size = 18)
axs1.set_ylabel('f(x)',size = 18)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs1.grid(True)
axs1.set_xticks([]), axs1.set_yticks([])
axs1.legend()
fig1.show()
fig1.savefig("Load_Prediction/SVR/Figures/SVR_Explained.pdf", bbox_inches='tight')

# Include a second figure to show the intuition behind the Kernel trick

# Define a sine wave with randomly distributed data points.
list2 = np.sin(x_axis)*5 + np.random.normal(0,1,100)
fig2, axs2=plt.subplots(1,2,figsize=(12,6))
axs2[0].scatter(x_axis, list2, color = "blue")
axs2[0].set_xlabel('x',size = 18)
axs2[0].set_ylabel('f(x)',size = 18)
axs2[0].set_xticks([]), axs2[0].set_yticks([])

axs2[1].scatter(x_axis, list1, label = "Datapoint",color = "blue")
axs2[1].plot(x_axis, x_axis, color = "red",label = "Prediction\nf(Φ(x))")
axs2[1].plot(x_axis, x_axis+a, '--', label = "Margin \n(+- ε)", color = "orange")
axs2[1].plot(x_axis, x_axis-a, '--', color = "orange")
axs2[1].set_xlabel('Φ(x)',size = 18)
axs2[1].set_ylabel('f(Φ(x))',size = 18)
axs2[1].set_xticks([]), axs2[1].set_yticks([])

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2[0].grid(True), axs2[1].grid(True)
axs2[1].legend()
fig2.show()
fig2.savefig("Load_Prediction/SVR/Figures/SVR_Explained_Kernel.pdf", bbox_inches='tight')
