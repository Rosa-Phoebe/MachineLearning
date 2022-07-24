# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 21:14:40 2022

Reference:
    https://github.com/gordicaleksa/get-started-with-JAX
    
    Build fully fledged Neural Networks.
    
    Gain knowledge necessary to build complex ML models(NNs) and train them in
    parallel on multiple devices.

@author: 
"""
import jax
import jax.numpy as jnp
import numpy as np

from jax import grad, jit, vmap, pmap
from jax import random

import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, NamedTuple
import functools



# JAX - functional programming paradigm. Impure functions are problematic.
# JAX - Pure Functions
g = 0 # state
# Access some external state in function which causes problems
def impure_uses_globals(x):
    return x + g
# JAX captures the value of global/state during the first run
print("First call:", jit(impure_uses_globals)(4.))

# Update global/state!
g = 10.
# Subsequent runs may silently use cached value of globals/state
print("Second call:", jit(impure_uses_globals)(5.))


# JAX - PRNG
# Which is not stateful in contrast to NumPy's PRNG
seed = 0
state = jax.random.PRNGKey(seed)
# The state is not saved internally
state1, state2 = jax.random.split(state)


# Let's now explicitly address and understand the problem of state! Why?
# Well, NNs love statefulness: model params, optimizer params, BatchNorm, etc.
# We've seen that JAX seems to have a problem with it.
class Counter:
    """ A simple counter. """
    def __init__(self):
        self.n = 0
        
    def count(self) -> int:
        self.n += 1
        return self.n
    
    def reset(self):
        self.n = 0
        
counter = Counter()
for _ in range(3):
    print(counter.count())

counter.reset()
fast_count = jax.jit(counter.count)
for _ in range(3):
    print(fast_count())

# Use `jaxpr` to understand why this is happening
from jax import make_jaxpr

counter.reset()
print(make_jaxpr(counter.count)())

# Solution:
#   Our counter state is implemented as a simple integer
CounterState = int
class CounterV2:
    def count(self, n: CounterState) -> Tuple[int, CounterState]:
        # Output and counter state
        return n + 1, n + 1
    
    def reset(self) -> CounterState:
        return 0

counter = CounterV2()
state = counter.reset()
fast_count = jax.jit(counter.count)

for _ in range(3):
    value, state = fast_count(state)
    print(value)



# Enter PyTree
# Why are gradients a problem in the first place?
# Still need to find a way to handle gradients when dealing with big NNs.
## PyTree basics
f = lambda x, y, z, w: x**2 + y**2 + z**2 + w**2
# JAX: .backward() is not that great
x, y, z, w = [1.] * 4
dfdx, dfdy, dfdz, dfdw = grad(f, argnums=(0, 1, 2, 3))(x, y, z, w)
print(dfdx, dfdy, dfdz, dfdw)

# But when we have 175B params like GPT3. We do have a better way.
# We want to, more naturally, wrap our params in some more complex data structures
# like dictionaries, etc.
# JAX knows how to deal with these! The answer is called a `PyTree`.
## A contrived example for pedagogical purposes
pytree_example = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have
for pytree in pytree_example:
    leaves = jax.tree_leaves(pytree)
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

# How do we manipulate PyTrees?
list_of_lists = [
    {'a': 3},
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]
# For single arg functions use `tree_map`
# `tree_map` iterates through leaves and applies the lambda function
print(jax.tree_map(lambda x: x*2, list_of_lists))

# PyTrees need to have same structure if we are to apply `tree_multimap`!
another_list_of_lists = list_of_lists
print(jax.tree_multimap(lambda x, y: x + y, list_of_lists, another_list_of_lists))

# ValueError: List arity mismatch: 5 != 4;
another_list_of_lists = deepcopy(list_of_lists)
another_list_of_lists.append([23])
print(jax.tree_multimap(lambda x, y: x + y, list_of_lists, another_list_of_lists))



# Training a toy MLP(multi-layer perceptron) model
def init_mlp_params(layer_widths):
    params = []
    # Allocate weights and biases (model parameters)
    # Notice: we're not using JAX's PRNG here
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
                 biases=np.ones(shape=(n_out,))
            )
        )
    return params

# Instantiate a single input - single output, 3 layer (2 hidden layers) deep MLP
params = init_mlp_params([1, 128, 128, 1])
# Another example of how we might use `tree_map`: verify shapes
jax.tree_map(lambda x: x.shape, params)



def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(jnp.dot(x, layer['weights']) + layer['biases'])
    
    return jnp.dot(x, last['weights']) + last['biases']

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2) # MSE loss

lr = 0.0001
# Notice how we do jit only at the highest level - XLA will have plenty of space to optimize
@jit 
def update(params, x, y):
    # Note that grads is a pytree with the same structure as params. grad is one
    # of many JAX functions that has built-in support for pytrees!
    grads = jax.grad(loss_fn)(params, x, y)
    
    # SGD update
    # For every leaf i.e. for every param of MLP
    return jax.tree_multimap(lambda p, g: p - lr*g, params, grads)

# Learn how to regress a parabola
xs = np.random.normal(size=(128, 1))
ys = xs ** 2

num_epochs = 5000
for _ in range(num_epochs):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label="Model prediction")
plt.legend()



# In order to be able to build NN libs and layers such as nn.Linear(PyTorch syntax)
# Custom PyTrees: This could be a linear layer, a conv layer or whatever
class MyContainer:
    """ A named container. """
    def __init__(self, name: str, a: int, b: int, c: int):
        self.name = name
        self.a = a
        self.b = b
        self.c = c
        
example_pytree = [MyContainer('Alice', 1, 2, 3), MyContainer('Bob', 4, 5, 6)]
leaves = jax.tree_leaves(example_pytree)
print(f"{repr(example_pytree):<45}\n has {len(leaves)} leaves:\n {leaves}")
    
# Need to define 2 functions (flatten/unflatten)
def flatten_MyContainer(container):
    """ Returns an iterable over container contents and aux data """
    flat_contents = [container.a, container.b, container.c]
    # We don't want name to appear as a child, so it is auxiliary data. auxiliary
    # data is usually a description of the structure of a node, e.g., the keys
    # of a dict -- anything that isn't a node's children.
    aux_data = container.name
    return flat_contents, aux_data

def unflatten_MyContainer(aux_data, flat_contents):
    """ Converts aux data and the flat contents into a MyContainer. """
    return MyContainer(aux_data, *flat_contents)

# Register a custom PyTree node
jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)
leaves = jax.tree_leaves(example_pytree)
print(f"{repr(example_pytree):<45}\n has {len(leaves)} leaves:\n {leaves}")



# How do we handle training it across multiple devices?
# Parallelism in JAX
# Parallelism in JAX is handled by another fundamental transform function: `pmap`
# """ Run it on Google Colab """
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
jax.devices()

# Use a simple running example (Single data)
x = np.arange(5) # signal
w = np.array([2., 3., 4.]) # window/kernel

# Implementation of 1D convolution/correlation
def convolve(w, x):
    output = []
    
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

result = convolve(w, x)
print(repr(result))


n_devices = jax.local_device_count()
print(f'Number of available devices: {n_devices}')
# Imagine we have a much heavier load(a "batch" of examples)
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)
print(xs.shape, ws.shape)
# First way to optimize this is to simply use `vmap`
vmap_result = jax.vmap(convolve)(ws, xs)
print(repr(vmap_result))

# Swap `vmap` for `pmap`, you are now running on multiple devices.
# No cross-device communication costs. Computations are done independently on each dev.
pmap_result = jax.pmap(convolve)(ws, xs)
print(repr(pmap_result))  # ShardedDeviceArray!

# Communication between devices
# We communicate across devices in order to normalize the outputs
def normalized_convolution(w, x):
    output = []
    
    for i in range(1, len(x) - 1):
        output.append(jnp.dot(x[i-1:i+2], w))
        
    output = jnp.array(output)
    
    # This is where communication happens
    return output / jax.lax.psum(output, axis_name='batch_dim')

# We don't have to manually broadcast w. Use `in_axes` args.
res_pmap = jax.pmap(normalized_convolution, axis_name='batch_dim', in_axes=(None, 0))(w, xs)
res_vmap = jax.vmap(normalized_convolution, axis_name='batch_dim', in_axes=(None, 0))(w, xs)
print(repr(res_pmap))
print(repr(res_vmap))

# Sometimes aside from grads we also need to return the loss value for logging
def sum_squared_error(x, y):
    return sum((x - y)**2)

x = jnp.arange(4, dtype=jnp.float32)
y = x + 0.1
# An efficient way to return both grads and loss value
jax.value_and_grad(sum_squared_error)(x, y)

# Sometimes loss function needs to return intermediate results
def sum_squared_error_with_aux(x, y):
    return sum((x - y)**2), x - y

#jax.grad(sum_squared_error_with_aux)(x, y)
jax.grad(sum_squared_error_with_aux, has_aux=True)(x, y)



# Training a very simple model in parallel!
# Train a model in parallel on multiple devices!
class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray

lr = 0.005

def init_model(rng):
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)

def forward(params, xs):
    return params.weight * xs + params.bias

def loss_fn(params, xs, ys):
    pred = forward(params, xs)
    return jnp.mean((pred - ys) ** 2) # MSE

@functools.partial(jax.pmap, axis_name='batch')
def update(params, xs, ys):
    # Compute the gradients on the given minibatch(individually on each device)
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)
    # Combine the gradient across all devices(by taking their mean)
    grads = jax.lax.pmean(grads, axis_name='batch')
    # Also combine the loss. Unnecessary for the update but useful for logging
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Each device performs its own SGD update, but since we start with the same
    # params and synchronise gradients, the params stay in sync on each device.
    new_params = jax.tree_multimap(lambda param, g: param - g*lr, params, grads)
    
    return new_params, loss

# Generate true data from y = w*x + b + noise
true_w, true_b = 2, -1
xs = np.random.normal(size=(128, 1))
noise = 0.5 * np.random.normal(size=(128, 1))
ys = xs * true_w + true_b + noise
plt.scatter(xs, ys)
plt.show()

# Initialize parameters and replicate across devices.
params = init_model(jax.random.PRNGKey(0))
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(lambda x: jnp.array([x] * n_devices), params)
# Prepare the data
def reshape_for_pmap(data, n_devices):
    return data.reshape(n_devices, data.shape[0]//n_devices, *data.shape[1:])

x_parallel = reshape_for_pmap(xs, n_devices)
y_parallel = reshape_for_pmap(ys, n_devices)
print(x_parallel.shape, y_parallel.shape)

def type_after_update(name, obj):
    print(f"after first `update()`, `{name}` is a {type(obj)}")

# Actual training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # This is where the params and data gets communicated to devices
    replicated_params, loss = update(replicated_params, x_parallel, y_parallel)
    
    # `replicated_params` and `loss` are now both ShardedDeviceArrays, indicating
    # that they're on the devices.
    # x/y_parallel remains a NumPy array on the host (simulating data streaming)
    if epoch == 0:
        type_after_update('replicated_params.weight', replicated_params.weight)
        type_after_update('loss', loss)
        type_after_update('x_parallel', x_parallel)
    
    if epoch % 100 == 0:
        print(loss.shape)
        print(f"Step {epoch:3d}, loss: {loss[0]:.3f}")

# Like the loss, the leaves of params have an extra leading dimension, 
# so we take the params from the first device.
params = jax.device_get(jax.tree_map(lambda x: x[0], replicated_params))

plt.scatter(xs, ys)
plt.plot(xs, forward(params, xs), c='red', label='Model Prediction')
plt.legend()
plt.show()