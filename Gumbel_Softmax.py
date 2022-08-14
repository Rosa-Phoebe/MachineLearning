# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:00:29 2022

@author: 
    
Reference:
    http://amid.fish/humble-gumbel
"""
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 4)


"""
    Gumbel-Softmax Distribution
    
    Why Gumbel-softmax distribution is important and how it is used?
"""
# Gumbel distribution
# Gumbel distribution is typically used to model the maximum of a set of
# independent samples.
# Toy example: Let's say you want to quantify how much ice cream you eat per day
#              . Assume hunger for ice cream is `normally-distributed` with mean
#              of 5. You record hunger 100 times a day for 10,000 days.
# Result: The distribution of maximum daily hunger values is Gumbel-distributed.
mean_hunger = 5
samples_per_day = 100
n_days = 10000
samples = np.random.normal(loc=mean_hunger, size=(n_days, samples_per_day))
daily_maxes = np.max(samples, axis=1) # (10000,)


# The Gumbel distribution is a probability distribution with density function
def gumbel_pdf(prob, loc, scale):
    z = (prob - loc) / scale
    return np.exp(-z - np.exp(-z)) / scale

def plot_maxes(daily_maxes):
    # probs: (1000,), hungers: (1001,)
    probs, hungers, _ = plt.hist(daily_maxes, density=True, stacked=True, bins=1000)
    plt.xlabel("Hunger")
    plt.ylabel("Probability of hunger being daily maximum")
    
    (loc, scale), _ = curve_fit(gumbel_pdf, hungers[:-1], probs)
    plt.plot(hungers, gumbel_pdf(hungers, loc, scale))
    
plt.figure()
plot_maxes(daily_maxes)



# The Gumbel-max trick
# What does the Gumbel distribution have to do with sampling from a categorical
# distribution?
n_cats = 7
cats = np.arange(n_cats)

probs = np.random.randint(low=1, high=20, size=n_cats)
probs = probs / sum(probs) # cats probability
logits = np.log(probs)

def plot_probs():
    plt.bar(cats, probs)
    plt.xlabel("Category")
    plt.ylabel("Probability")

plt.figure()
plot_probs()    


n_samples = 1000
def plot_estimated_probs(samples):
    n_cats = np.max(samples) + 1
    estd_probs, _, _ = plt.hist(samples, bins=np.arange(n_cats + 1), align='left',
                                edgecolor='White', density=True, stacked=True)
    plt.xlabel("Category")
    plt.ylabel("Estimated probability")
    return estd_probs

def print_probs(probs):
    print(" ".join(["{:.2f}"] * len(probs)).format(*probs))

# Generates a random sample from a given 1-D array
samples = np.random.choice(cats, p=probs, size=n_samples)
plt.figure()
plt.subplot(1, 2, 1)
plot_probs()
plt.subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
plt.tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

# The above original and estimated probabilities seems look good. The trick will
# be on the following case. (Introduce Gumbel distribution)
#
# Sampling with different types of noise
# 1. Uniform noise
def sample(logits):
    noise = np.random.uniform(size=(len(logits)))
    sample = np.argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

plt.figure()
plt.subplot(1, 2, 1)
plot_probs()
plt.subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
plt.tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

# The above uniform noise seems to capture the modes of the distribution but
# distorted. It also completely misses out all the other categories.


# 2. Normal noise
def sample(logits):
    noise = np.random.normal(size=len(logits))
    sample = np.argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

plt.figure()
plt.subplot(1, 2, 1)
plot_probs()
plt.subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
plt.tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

# Normal noise seems to do a better job of capturing the full range of categories
# , but still distorts the probabilities. We're getting closer though. Maybe
# there is a special kind of noise that gives the right results.
#
#
#
# It turns out the `special kind of noise` happens to be noise from a Gumbel
# distribution!
# 3. Gumbel noise
def sample(logits):
    noise = np.random.gumbel(size=len(logits))
    sample = np.argmax(logits + noise)
    return sample

samples = [sample(logits) for _ in range(n_samples)]

plt.figure()
plt.subplot(1, 2, 1)
plot_probs()
plt.subplot(1, 2, 2)
estd_probs = plot_estimated_probs(samples)
plt.tight_layout()

print("Original probabilities:\t\t", end="")
print_probs(probs)
print("Estimated probabilities:\t", end="")
print_probs(estd_probs)

# With Gumbel noise, we get exactly the right probabilities! Why is it that Gumbel
# noise happens to be exactly the right kind of noise to make this work?



# Generate Gumbel noise
# What if we only access to a uniform noise generator? It turns out we can still
# generate Gumbel noise by starting with uniform noise and then taking the negative
# log twice.
numpy_gumbel = np.random.gumbel(size=n_samples)
manual_gumbel = -np.log(-np.log(np.random.uniform(size=n_samples)))
plt.figure()
plt.subplot(1, 2, 1)
plt.hist(numpy_gumbel, bins=50, density=True, stacked=True)
plt.ylabel("Probability")
plt.xlabel("numpy Gumbel")
plt.subplot(1, 2, 2)
plt.hist(manual_gumbel, bins=50, density=True, stacked=True)
plt.xlabel("Gumbel from unform noise")

# Why sample with Gumbel noise? Sampling this way is called the `Gumbel-max trick`.
# The Gumbel-softmax trick: The Gumbel-max trick produces a sample by adding noise
# to the logits then taking the argmax of the resulting vector.

# But what if we approximate the argmax by a softmax? Then something really
# interesting happens: we have a chain of operations that's fully differentiable
# . We have differentiable sampling operator.