# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:01:18 2022

Reference:
    https://jaketae.github.io/study/gibbs-sampling/
    
    
    Sample from a bivariate Gaussian distribution

@author: 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)



# This function shows the gist of how Gibbs sampling works
def gibbs_sampler(mus, sigmas, n_iter=5000):
    samples = []
    y = mus[1]
    for _ in range(n_iter):
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples.append([x, y])
    return samples

# Derive the equation for conditional probability distributions of bivariate Gaussian
# Note: Two functions are symmetrical which is expected given that this is a
#       bivariate distribution. These functions simulate a conditional distribution
#       , where given a value of one random variable, we can sample the value of
#       the other. This is core mechanism by which we will be sampling from the
#       joint probability distribution using the Gibbs sampling algorithm.
def p_x_given_y(y, mus, sigmas):
    mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
    sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
    return np.random.normal(mu, sigma)

def p_y_given_x(x, mus, sigmas):
    mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
    sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
    return np.random.normal(mu, sigma)
    
# Initialize parameters for distribution and test the sampler
mus = np.asarray([5, 5]) # numpy.asarray: Convert the input to an array
sigmas = np.asarray([[1, .9], [.9, 1]])
samples = gibbs_sampler(mus, sigmas)
print(samples[:5])

# For purposes of demonstrating the implications of burn-in and discard first
# 100 values that were sampled.
burn = 100
x, y = zip(*samples[burn:])
sns.jointplot(x, y, kind='hex')
plt.show()

# Result produced from direct sampling
samples = np.random.multivariate_normal(mus, sigmas, 10000)
sns.jointplot(samples[:, 0], samples[:, 1], kind='kde')
plt.show()



# Conclusion:
# Even if we can't directly sample from the distribution, if we have access to
# conditional distributions, we can still achieve an asymptotically similar
# result.