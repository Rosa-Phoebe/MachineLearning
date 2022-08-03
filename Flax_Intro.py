# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 23:18:13 2022

Reference:
    https://github.com/gordicaleksa/get-started-with-JAX
    
    
    Flax was "designed for flexibility" hence the name:
        Flexibility + JAX -> Flax
    What Flax offers is high performance and flexibility (similarly to JAX)

@author:
"""
import numpy as np

import jax
from jax import lax, random, numpy as jnp

# NN lab built on top of JAX developed by Google Research
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax.training import train_state




# Flax doesn't have its own data loading functions - Use PyTorch dataloaders
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# Python libs
from typing import Sequence, Callable

from inspect import signature


# Start with simplest model possible: single feed-forward layer
model = nn.Dense(features=5)
# All of Flax NN layers inherit from `Module` class
# <class 'flax.linen.module.Module'>
print(nn.Dense.__bases__)


# How can we do inference with simple model? Two steps: init and apply.
# Step 1: init
seed = 23
key1, key2 = random.split(random.PRNGKey(seed))
# Create a dummy input, a 10-dimensional random vector
x = random.normal(key1, (10,))


# Initialization call - gives us the actual model weights
# Note1: automatic shape inference
# Note2: immutable structure (hence FrozenDict)
# Note3: init_with_output if you care, for whatever reason, about the output here
y, params = model.init_with_output(key2, x) # Care about the output
print(y)
print(jax.tree_map(lambda x: x.shape, params)) # Check output(Note1,2,3)

# Step 2: apply
# This is how you run prediction in Flax, state is external!
y = model.apply(params, x)
print(y) # This one is equal above one!



# A toy example - training a linear regression model
# Define a toy dataset
n_samples = 150
x_dim = 2
y_dim = 1
noise_amplitude = 0.1

# Generate (random) ground truth W and b
key, W_key, b_key = random.split(random.PRNGKey(seed), num=3)
W = random.normal(W_key, (x_dim, y_dim)) # weight
b = random.normal(b_key, (y_dim,)) # bias

# This is structure that Flax expects
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise
key, x_key, noise_key = random.split(key, num=3)
xs = random.normal(x_key, (n_samples, x_dim))
ys = jnp.dot(xs, W) + b
ys += noise_amplitude * random.normal(noise_key, (n_samples, y_dim))
print(f'xs shape = {xs.shape}; ys shape = {ys.shape}')

# Visualize data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
assert xs.shape[-1] == 2 and ys.shape[-1] == 1
ax.scatter(xs[:, 0], xs[:, 1], zs=ys)


def make_mse_loss(xs, ys):
    def mse_loss(params): # pure function
        def squared_error(x, y):
            pred = model.apply(params, x)
            return jnp.inner(y-pred, y-pred) / 2.0
        
        # Batched version via vmap
        return jnp.mean(jax.vmap(squared_error)(xs, ys), axis=0)
    
    return jax.jit(mse_loss)

mse_loss = make_mse_loss(xs, ys)
value_and_grad_fn = jax.value_and_grad(mse_loss)

# Simple feed-forward layer
model = nn.Dense(features=y_dim)
params = model.init(key, xs)
print(f'Initial params = {params}')

lr = 0.3
epochs = 20
log_period_epoch = 5

print('-' * 50)
for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params)
    params = jax.tree_multimap(lambda p, g: p - lr * g, params, grads)

    if epoch % log_period_epoch == 0:
        print(f'epoch {epoch}, loss = {loss}')

print('-' * 50)
print(f'Learned params = {params}')
print(f'Gt params = {true_params}')



# Create custom models
class MLP(nn.Module):  # `nn.Module` is Python's dataclass
    num_neurons_per_layer: Sequence[int]
    
    def setup(self):
        self.layers = [nn.Dense(n) for n in self.num_neurons_per_layer]
        
    def __call__(self, x):
        activation = x
        for i, layer in enumerate(self.layers):
            activation = layer(activation)
            if i != len(self.layers) - 1:
                activation = nn.relu(activation)
        return activation
    
x_key, init_key = random.split(random.PRNGKey(seed))
# Define an MLP model
model = MLP(num_neurons_per_layer=[16, 8, 1])
x = random.uniform(x_key, (4, 4)) # dummy input
params = model.init(init_key, x)  # initialize via init
y = model.apply(params, x) # do forward pass via apply

print(jax.tree_map(jnp.shape, params))
print(f'Output: {y}')


# Dive deeper and understand how the `nn.Dense` module is designed
## Introducing "param"
class MyDenseImp(nn.Module):
    num_neurons: int
    weight_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', self.weight_init, (x.shape[-1], self.num_neurons))
        bias = self.param('bias', self.bias_init, (self.num_neurons,))
        return jnp.dot(x, weight) + bias

# You can see it expects a PRNG key and it is passed implicitly through the init
# fn (same for zeros)
# (key, shape, dtype=<class 'jax.numpy.float64'>)
print(signature(nn.initializers.lecun_normal()))