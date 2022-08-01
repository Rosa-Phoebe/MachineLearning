# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:28:38 2022

Reference:
    JAX Crash Course - Accelerating Machine Learning code
    
Run it on Google Colab

@author: 
"""

"""
    What is JAX?
    
    JAX is `Autograd` and `XLA`, brought together for high-performance numerical
    computing and machine learning research. It provides composable transformations
    of `Python+NumPy` programs: differentiate, vectorize, parallelize, Just-In-
    Time compile to GPU/TPU and more.
    
    In simpler words: JAX is NumPy on the CPU, GPU and TPU with great automatic
    differentiation for high-performance machine learning research.
    
    XLA: Accelerated Linear Algebra, lies at the foundation of what makes JAX
         so powerful. Developed by Google, XLA is a domain-specific, graph-based
         , just-in-time compiler for linear algebra. It significantly improves
         execution speed and lowers memory usage by fusing low-level operations.
         
    Outline
    
    Speed comparison
    Replacement for NumPy
    Speed up computations with jit()
    Automatic Differentiation with grad()
    Automatic Vectorization with vmap()
"""

# Speed comparison
# Huge performance boost when running on GPU/TPU but even on CPU!
import numpy as np

import jax.numpy as jnp
from jax import jit
from jax import grad
from jax import jacfwd
from jax import vmap

import matplotlib.pyplot as plt



def fn(x):
    return x + x*x + x*x*x

x = np.random.randn(10000, 10000).astype(dtype='float32')
# -n: how many times to execute ‘statement’
%timeit -n5 fn(x)


jax_fn = jit(fn)
x = jnp.array(x)
# Use `block_until_ready` because JAX uses asynchronous execution by default
%timeit -n5 jax_fn(x).block_until_ready()



### Drop-in Replacement for NumPy: Use this alone can bring a nice performance
### boost on the GPU/TPU.
# JAX numpy can often be used as drop-in replacement for numpy since the API is
# almost identical. One useful feature of JAX is that the same code can be run
# on different backends - CPU, GPU and TPU.
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)

# Big difference: JAX arrays are immutable!
x = np.arange(10)
x[0] = 10
print(x)

# '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item
# assignment. JAX arrays are immutable.
x = jnp.arange(10)
x[0] = 10 # Error



### Speed up computations with `jit()`
# Just-in-time or JIT compilation, is a method of executing code that lies
# between interpretation and ahead-of-time(AoT) compilation. The important fact
# is that a JIT-compiler will compile code at runtime into a fast executable
# (at the cost of a slower first run). 
def fn(x):
    return x + x*x + x*x*x

x_np = np.random.randn(5000, 5000).astype(dtype='float32')
%timeit -n5 fn(x_np)

x_jnp = jnp.array(x_np)
%timeit -n5 fn(x_jnp).block_until_ready()

jitted = jit(fn)
%timeit -n5 jitted(x_jnp).block_until_ready()

# Use as decorator
@jit
def fn_jitted(x):
    return x + x*x + x*x*x

%timeit -n5 fn_jitted(x_jnp).block_until_ready()

# How does JIT works?
# JIT and other JAX transforms work by tracing a function to determine its
# effect on inputs of a specific shape and type.
@jit
def f(x, y):
    print("Running f():")
    print(f"  x = {x}")
    print(f"  y = {y}")
    result = jnp.dot(x + 1, y + 1)
    print(f"  result = {result}")
    return result

x = np.random.randn(3, 4)
y = np.random.randn(4)

# Notice that the `print` statements execute, but rather than printing the data
# we passed to the function, though, it prints tracer objects that stand-in for
# them.
#
# These tracer objects are what `jax.jit` uses to extract the sequence of operations
# specified by the function. Basic tracers are stand-ins that encode the shape
# and dtype of the arrays, but are agnostic to the values. This recorded sequence
# of computations can then be efficiently applied with XLA to new inputs with
# the same shape and dtype without having to re-execute the Python code.
f(x, y)

# When we call the compiled function again on matching inputs, no re-compilation
# is required and nothing is printed because the result is computed in compiled
# XLA rather than in Python.
f(x, y)

# Limitations of JIT
# Important: Because JIT compilation is done without information on the content
#            of the array, control flow statements in the function cannot depend
#            on traced values. But You can have JIT-able functions with control
#            flow, but you have to rephrase them with `jax.lax`, ie `lax.cond`
#            instead of `if/else` etc.
def f(x):
    if x > 0:
        return x
    else:
        return 2 * x
    
f_jit = jit(f)
f_jit(10) # Should raise an error



### Automatic Differentiation with `grad()`
# Very simple to calculate gradients and second/third order gradients etc.
# The gradients follow more the underlying math rather than using backpropagation
# in other Deep Learning libraries -> It can be much faster with JAX!
#
# `grad()` works for scalar-valued function meaning a function which maps
# scalars/vectors to scalars.
def f(x):
    return x**3 + 2*x**2 - 3*x + 1

dfdx = grad(f)
d2fdx = grad(grad(f))

print(f"x = 1.0")
print(f"f  (x) = {f(1.)}")
print(f"f' (x) = 3*x^2 + 4x - 3 = {dfdx(1.)}")
print(f"f''(x) = 6x + 4 = {d2fdx(1.)}")

# Another example
def rectified_cube(x):
    r = 1
    
    if x < 0:
        for i in range(3):
            r *= x
        r = -r
    else:
        for i in range(3):
            r *= x
    return r

gradient_function = grad(rectified_cube)
print(f"x = 2   f(x) = {rectified_cube(2.)}   f'(x) =  3*x^2 = {gradient_function(2.)}")
print(f"x = -3  f(x) = {rectified_cube(-3.)}  f'(x) = -3*x^2 = {gradient_function(-3.)}")


# Jacobian and Hessian for Vector-Valued Functions
# For vector-valued functions which map vectors to vectors, the analogue to the
# gradient is the Jacobian. With the function transformations `jacfwd()` and
# `jacrev()`, corresponding to forward mode differentiation and reverse mode
# differentiation.
def mapping(v): # vector-valued function
    x = v[0]
    y = v[1]
    z = v[2]
    return jnp.array([x*x, y*z])

# 3 inputs, 2 outputs
# [d/dx x^2 , d/dy x^2, d/dz x^2]
# [d/dx y*z , d/dy y*z, d/dz y*z]

# [2*x , 0, 0]
# [0 , z, y]

# f(x, y, z) = x^2, y*z

f = jacfwd(mapping)
v = jnp.array([4., 5., 9.])
print(f(v))

def hessian(f):
    return jacfwd(grad(f))

def f(x):
    return jnp.dot(x, x)

hessian(f)(jnp.array([1., 2., 3.]))



### Automatic Vectorization with `vmap()`
# JAX's transform: Vectorization via vmap(the vectorizing map). It has the familiar
# semantics of mapping a function along array axes, but instead of keeping the
# loop on the outside, it pushed the loop down into a function's primitive
# operations for better performance.
x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

print(convolve(x, w))

# Suppose we would like to apply this function to a batch of weights w to a
# batch of vectors x.
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

# The most naive option would be to simply loop over the batch in Python
def manually_batched_convolve(xs, ws):
    output = []
    
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)

print(manually_batched_convolve(xs, ws))

# use vmap
auto_batch_convolve = vmap(convolve)
print(auto_batch_convolve(xs, ws))



# Example Training Loop with JAX
xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise

plt.scatter(xs, ys)

def model(theta, x):
    """Computes wx + b on a batch of input x."""
    w, b = theta
    return w * x + b

def loss_fn(theta, x, y):
    prediction = model(theta, x)
    return jnp.mean((prediction - y)**2)

def update(theta, x, y, lr=0.1):
    return theta - lr * grad(loss_fn)(theta, x, y)


theta = jnp.array([1., 1.])

for _ in range(1000):
    theta = update(theta, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, model(theta, xs))

w, b = theta
print(f"w: {w:<.2f}, b: {b:<.2f}")
print(f"f(x) = {w:<.2f} * x {b:<.2f}")