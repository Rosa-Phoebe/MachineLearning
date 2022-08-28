# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:24:17 2022

Reference:
    Machine Learning & Simulation

@author: 
"""

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

"""
    Often, one is not interested in the full Jacobian matrix of a vector-valued
    function, but its matrix multiplication with a vector. This operation is
    represented by the jvp intrinsic in the JAX deep learning framework.
    
    JAX as well as others Automatic Differentiation frameworks, provides an
    intrinsic called jvp to perform this Jacobian-Vector-Product highly efficiently
    by employing forward-mode AD directly through the computation graph of the
    matrix-vector product.

    Jacobian-Vector product is a primitive for forward mode automatic differentiation
"""
def f(x):
    # A vector-valued function takes input - a four dimensional vector space x
    # and maps that one to a three-dimensional output
    u = jnp.array([
        x[0]**6 * x[1]**4 * x[2]**9 * x[3]**2,
        x[0]**2 * x[1]**3 * x[2]**5 * x[3]**3,
        x[0]**5 * x[1]**7 * x[2]**7 * x[3]**6,
    ])
    return u

# Create an evaluation point vector
evaluation_point = jnp.array([1.0, 0.5, 1.5, 2.0])

# Query this function at the evaluation point
f(evaluation_point)

# Jacobian is a matrix that is given by the derivative of the function with
# respect to its input. Jacobian: df/dx.
# Use jax forward mode automatic differentiation in order to obtain it.
full_jacobian = jax.jacfwd(f)(evaluation_point)

# Jacobian-Vector product df/dx @ v
multiplication_point = jnp.array([0.2, 0.3, 0.4, 0.8])

# The essence of a Jacobian vector product
# First obtain full and dense Jacobian then perform matrix multiplication
full_jacobian @ multiplication_point

# Use Jax automatic differentiation to automatically obtain the result without
# explicitly computing the Jacobian.
# Use a JAX function to obtain the result of the jvp without explicitly computing
# the Jacobian matrix
## `primals` are points at which the Jacobian shall be evaluated
## `tangents` is the vector with which we want to multiply
## These names arise from so-called differentiable geometry
# jax.jvp(f, primals, tangents)
# Return: an evaluation of function f, an evaluation of Jacobian vector product
f_evaluated, jvp_evaluated = jax.jvp(f, (evaluation_point,), (multiplication_point))



"""
    Naive Jacobian-Vector product vs JAX.jvp (Benchmark comparison)
    
    The jvp intrinsic of the JAX deep learning framework works by directly
    performing forward-mode automatic differentiation over the computation graph
    of the matrix-vector product. Naive implementation is computing the full
    dense Jacobian matrix.
    
    The Jabobian matrices of many vector-valued functions exhibit a sparsity
    structure that is caused by the original functions having a small locality.
    
"""
# A vector-valued function with a small locality
# Input - 100-dimensional vector, output - 100-dimensional vector
def f(x):
    u = x**3 - jnp.roll(x, -1)**2 - jnp.roll(x, +1)**2
    return u

evaluation_point = jax.random.normal(jax.random.PRNGKey(42), (100,)) # 100-dimensional vector
print(evaluation_point.shape)
print(f(evaluation_point))
print(f(evaluation_point).shape)

# Use automatic differentiation to obtain full Jacobian
jac = jax.jacfwd(f)(evaluation_point)
print(jac)
print(jac.shape)

# Use matplotlib to quickly visualize the structure of this matrix
# `matplotlib.pyplot.spy`: Plot the sparsity pattern of a 2D array. This visualizes
# the non-zero values of the array.
# We see the sparsity structure of the Jacobian matrix
plt.spy(jac)
plt.show()

# In order to make really fair comparison, employ just-in-time compilation of Jax
@jax.jit
def naive_jvp(primal, tangent):
    # `primal` and `tangent` are just differentiable geometry terms for evaluation
    # point and multiplication point.
    full_jacobian = jax.jacfwd(f)(primal)
    jvp_result = full_jacobian @ tangent
    return jvp_result

@jax.jit
def clever_jvp(primal, tangent):
    _, jvp_result = jax.jvp(f, (primal, ), (tangent, ))
    return jvp_result

multiplication_point = jax.random.normal(jax.random.PRNGKey(43), (100, ))
naive_jvp(evaluation_point, multiplication_point)
clever_jvp(evaluation_point, multiplication_point)
    
# %timeit naive_jvp(evaluation_point, multiplication_point)
# %timeit clever_jvp(evaluation_point, multiplication_point)



"""
    What is a Vector-Jacobian product(VJP) in JAX?
    
    Reverse-mode automatic differentiation is the essential ingredient to training
    NNets. (`Pullback` also called `VJP` in the JAX deep learning framework)
    
    The `VJP` intrinsics of JAX allow evaluating the left-multiplication of a
    vector to the Jacobian matrix of a vector-valued function at high efficiency.
    This is particularly necessary for sparse Jacobians of functions with small
    locality.
"""
def f(x):
    # A vector-valued function takes input - a four dimensional vector space x
    # and maps that one to a three-dimensional output
    u = jnp.array([
        x[0]**6 * x[1]**4 * x[2]**9 * x[3]**2,
        x[0]**2 * x[1]**3 * x[2]**5 * x[3]**3,
        x[0]**5 * x[1]**7 * x[2]**7 * x[3]**6,
    ])
    return u

evaluation_point = jnp.array([1.0, 0.5, 1.5, 2.0])
f(evaluation_point)

# Jacobian df/dx
full_jacobian = jax.jacrev(f)(evaluation_point)

# Vector-Jacobian product: a.T @ df/dx
left_multiplication_point = jnp.array([0.5, 0.8, 1.0])
left_multiplication_point.T @ full_jacobian
print(full_jacobian.shape)

# Similar to the Jacobian-vector product, oftentimes we're not interested in the
# full Jacobian but only in the result of vector-Jacobian product.
# Using a JAX function to obtain the vjp without explicitly computing the full
# Jacobian matrix.
# For `f_evaluated`: whenever we do reverse-mode automatic differentiation, we
# need one forward pass through the actual function that we want to differentiate.
# What is happening under the hood in `jax.vjp`?
# => When call the vjp, then jax is tracing the execution of f at a particular
#    evaluation point and it's saving intermediate values which it can then use
#    with a call to `vjp_function` to compute derivative information.
f_evaluated, vjp_function = jax.vjp(f, evaluation_point)
# `vjp_function(cotangent)`: cotangent is just a term from differentiable
# geometry. 
vjp_function(left_multiplication_point)



"""
    Naive vector-Jacobian product vs JAX.vjp. Benchmark comparison
    
    Vector-Jocobian products evaluate the left multiplication of a (cotangent)
    vector to a Jacobian matrix. The intrinsic of the JAX deep learning framework
    is faster than a naive implementation for sparse Jacobians.
    
    Sparse Jacobians are common in Scientific Computing. Unfortunately, JAX does
    not support sparse array storage formats or sparse Automatic Differentiation.
    Hence, obtaining a Jacobian matrix of such functions will allocate a dense
    array. This quickly becomes intractable/infeasible for higher dimensional
    problems due to the unnecessary computation and storage of many zeros.
    
    Luckily, often the full Jacobian is of no relevance but its effect on vectors
    that are either multiplied from the right (Jacobian-vector product) or from
    the left (vector-Jacobian product).
"""
# A vector-valued function with a small locality
# Input - 100-dimensional vector, output - 100-dimensional vector
def f(x):
    u = x**3 - jnp.roll(x, -1)**2 - jnp.roll(x, +1)**2
    return u

evaluation_point = jax.random.normal(jax.random.PRNGKey(42), (100, ))
f(evaluation_point)

# Use reverse node to obtain the full Jacobian on f
jac = jax.jacrev(f)(evaluation_point)
plt.spy(jac)
plt.show()


@jax.jit
def naive_vjp(primal, cotangent):
    # `cotangent` is a term from differentiable geometry but here we can think
    # of it pragmatically as the vector that is left multiplied to the Jacobian
    # matrix.
    full_jacobian = jax.jacrev(f)(primal)
    vjp_result = cotangent.T @ full_jacobian
    return vjp_result

@jax.jit
def clever_vjp(primal, cotangent):
    # `vjp_function` performs vector jocobian product
    f_evaluated, vjp_function = jax.vjp(f, primal)
    # The comma here because this vjp will return a tuple and it will only have
    # one entry and that's because f only has one input and so unpack the tuple
    vjp_result, = vjp_function(cotangent)
    return vjp_result
    
left_multiplication_point = jax.random.normal(jax.random.PRNGKey(43), (100,))
naive_vjp(evaluation_point, left_multiplication_point)
clever_vjp(evaluation_point, left_multiplication_point)

# %timeit naive_vjp(evaluation_point, left_multiplication_point)
# %timeit clever_vjp(evaluation_point, left_multiplication_point)