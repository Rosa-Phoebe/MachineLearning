# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 19:52:46 2022

Reference:
    https://www.deepmind.com/blog/using-jax-to-accelerate-our-research
    https://github.com/gordicaleksa/get-started-with-JAX

@author: 
"""

"""
    `JAX` ecosystem is becoming an increasingly popular alternative to `PyTorch`
    and `TensorFlow`.
    
    JAX is Autograd and XLA, brought together for high-performance numerical
    computing and machine learning research. Maybe JAX stands for JIT(J),
    Autograd(A), XLA(X), which is essentially an abbreviation for a bunch of
    abbreviations.
    
    I> Warming up with JAX
    II> Going deeper
"""

# JAX's syntax is (for the most part) same as NumPy's!
# There is also SciPy API support (jax.scipy)
import jax.numpy as jnp
import numpy as np

# Special transform functions
from jax import grad, jit, vmap, pmap

# JAX's low level API
# lax is just an anagram for XLA.
from jax import lax
from jax import random
from jax import device_put

import matplotlib.pyplot as plt



# Fact 1: JAX's syntax is remarkably similar to NumPy's
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)



# Fact 2: JAX arrays are immutable! (embrace functional programming paradigms)
size = 10
index = 0
value = 23
## NumPy: mutable arrays
x = np.arange(size)
print(x)
x[index] = value
print(x)

## JAX: immutable arrays
x = jnp.arange(size)
print(x)
# '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item
# assignment
x[index] = value
print(x)



# Fact 3: JAX handles random numbers differently
seed = 0
key = random.PRNGKey(seed)
x = random.normal(key, (10,))
print(type(x), x)



# Fact 4: JAX is AI accelerator agnostic. Same code runs everywhere!
size = 3000

## Data is automagically pushed to the AI accelerator! (DeviceArray structure)
## No more need for ".to(device)" (PyTorch syntax)
x_jnp = random.normal(key, (size, size), dtype=jnp.float32)
x_np = np.random.normal(size=(size, size)).astype(np.float32)

## block_until_ready() -> asynchronous dispatch
## Note: Add time module
jnp.dot(x_jnp, x_jnp.T).block_until_ready() # 1) on GPU -fast
np.dot(x_np, x_np.T) # 2) Work on CPU - slow (NumPy only works with CPUs)
jnp.dot(x_np, x_np.T).block_until_ready() # 3) on GPU with transfer overhead

x_np_device = device_put(x_np) # push NumPy explicitly to GPU
jnp.dot(x_np_device, x_np_device.T).block_until_ready() # Same as 1)



""" JAX transform functions """
# Transformation function: `jit()` compiles your functions using XLA and caches them
## Simple helper visualization function
def visualize_fn(fn, l=-10, r=10, n=1000):
    x = np.linspace(l, r, num=n)
    y = fn(x)
    plt.plot(x, y)
    plt.show()
    
## Define a function: SELU activation function
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

## Visualize SELU (just for understanding, it's always a good ideas to visualize
## stuff)
visualize_fn(selu)

## Transform 'SELU': Compile it
selu_jit = jit(selu)
## Benchmark non-jit vs jit version
data = random.normal(key, (1000000,))
print('non-jit version:')
selu(data).block_until_ready() # Add time module
print('jit version:')
selu_jit(data).block_until_ready()



# Transformation function: `grad()`
# Differentiation can be: manual, symbolic, numeric, automatic!
## First example - Automatic diff(autodiff)
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x = jnp.arange(3.)
loss = sum_logistic # rename it to give it some semantics
# By default `grad` calculates derivative of a fn w.r.t. 1st parameter!
grad_loss = grad(loss)
print(grad_loss(x))

## Numeric diff (to double check that autodiff works correctly)
def finite_differences(f, x):
    eps = 1e-3
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))])

print(finite_differences(loss, x))

## Second example (automatic diff)
x = 1.
f = lambda x: x**2 + x + 4 # Simple 2nd order polynomial fn
visualize_fn(f, l=-1, r=2, n=100)

dfdx = grad(f) # 2*x + 1
d2fdx = grad(dfdx) # 2
d3fdx = grad(d2fdx)

print(f(x), dfdx(x), d2fdx(x), d3fdx(x))



# JAX autodiff engine is very powerful ("advanced" example)
# There some questions here???
from jax import jacfwd, jacrev

f = lambda x, y: x**2 + y**2   # simple paraboloid
## Jacobian
## df/dx = 2x, df/dy = 2y
## J = [df/dx, df/dy]

## Hessian
## d2f/dx = 2, d2f/dy = 2
## d2f/dxdy = 0, d2f/dy = 0
## H = [[d2f/dx, d2f/dxdy], [d2f/dydx, d2f/dy]]

def hessian(f):
    return jit(jacfwd(jacrev(f, argnums=(0, 1)), argnums=(0, 1)))
print(f'Jacobian = {jacrev(f, argnums=(0, 1))(1., 1.)}')
print(f'Full Hessian = {hessian(f)(1., 1.)}')

## Edge case |x|, how does JAX handles it?
f = lambda x: abs(x)
visualize_fn(f)
print(f(-1), f(1))
dfdx = grad(f)
print(dfdx(0.), dfdx(-1.))



# Transformation function: `vmap()` (Need more understanding???)
W = random.normal(key, (150, 100)) # weights of a linear NN layer
batched_x = random.normal(key, (10, 100)) # a batch of 10 flattened images
def apply_matrix(x):
    return jnp.dot(W, x)

def naively_batched_apply_matrix(batched_x):
    return jnp.stack([apply_matrix(x) for x in batched_x])

print('Naively batched')
naively_batched_apply_matrix(batched_x).block_until_ready() # Add time module

@jit
def batched_apply_matrix(batched_x):
    return jnp.dot(batched_x, W.T)

print('Manually batched')
batched_apply_matrix(batched_x).block_until_ready() # Add time module

@jit # Note: we can arbitrarily compose JAX transforms!
def vmap_batched_apply_matrix(batched_x):
    return vmap(apply_matrix)(batched_x)

print('Auto-vectorized with vmap')
vmap_batched_apply_matrix(batched_x).block_until_ready() # Add time module



""" Going deeper """
# JAX API structure: NumPy <-> lax <-> XLA.
# lax API is stricter and more powerful and it's Python wrapper around XLA

# Example 1: lax is stricter
print(jnp.add(1, 1.0)) # jax.numpy API implicitly promotes mixed types
## lax.add requires arguments to have the same dtypes, got int32, float32.
print(lax.add(1, 1.0)) # jax.lax API requires explicit type promotion



# XLA: https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution
# Need more understanding???
# Example 2: lax is more powerful (but as a tradeoff less user-friendly)
x = jnp.array([1, 2, 1])
y = jnp.ones(10)

# NumPy API
result1 = jnp.convolve(x, y)

# lax API
result2 = lax.conv_general_dilated(
    x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
    y.reshape(1, 1, 10),
    window_strides=(1,),
    padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy

print(result1)
print(result2[0][0])
assert np.allclose(result1, result2[0][0], atol=1E-6)



# How does JIT actually work?
# Why JIT? -> jitted functions are much faster.
def norm(X):
    X = X - X.mean(0)
    return X/X.std(0)

norm_compiled = jit(norm)
X = random.normal(key, (1000, 100), dtype=jnp.float32)
assert np.allclose(norm(X), norm_compiled(X), atol=1E-6)
norm(X).block_until_ready() # Add time module
norm_compiled(X).block_until_ready() # Add time module

# Add more content for JIT




































