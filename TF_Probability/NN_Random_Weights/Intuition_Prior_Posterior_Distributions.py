import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

mu_prior = 0
variance_prior = 1.5
sigma_prior = math.sqrt(variance_prior)

mu_evidence = -3
variance_evidence = 0.75
sigma_evidence = math.sqrt(variance_evidence)

mu_posterior = -2
variance_posterior = 1
sigma_posterior = math.sqrt(variance_posterior)

x_axis = np.linspace(-9, 5, 100)
fig, axs=plt.subplots(1,1,figsize=(12,6))
axs.plot(x_axis, stats.norm.pdf(x_axis, mu_prior, sigma_prior), linewidth = 2, label = "Prior Distribution P(A)", color = "red")
axs.plot(x_axis, stats.norm.pdf(x_axis, mu_evidence, sigma_evidence), linewidth = 2, label = "Evidence P(B)", color = "blue")
axs.plot(x_axis, stats.norm.pdf(x_axis, mu_posterior, sigma_posterior), linewidth = 2, label = "Posterior Distribution P(A|B)", color = "orange")
#axs.grid(b=True, which='major'), axs.grid(b=True, which='minor',alpha = 0.2)
#axs.tick_params(axis = "both", labelsize = 12)
#axs.minorticks_on()
axs.set_xticks([]), axs.set_yticks([])
axs.legend(fontsize=14)
fig.show()
fig.savefig("TF_Probability/NN_Random_Weights/Figures/Visual_Intuition_Prior_and_Posterior.pdf", bbox_inches='tight')
