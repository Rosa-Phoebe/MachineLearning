# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:21:38 2022

    Reference:
        Quickstart with Optax
        
    `Optax` is a simple optimization library for `Jax`.

@author: 
"""


import jax.numpy as jnp
import jax
import optax
import functools


# Set up a simple linear model and a loss function.
# Note: `in_axes=(None, 0)` => mean that the first argument (here `params`) will
#       not be mapped, while the second argument (here `x`) will be mapped along
#       axis 0.
# jax.vmap(network, in_axes=(None, 0))(params, x)
@functools.partial(jax.vmap, in_axes=(None, 0))
def network(params, x):
    return jnp.dot(params, x)

# The loss function (L2 loss) comes from optax's common loss function via
# `optax.l2_loss()`.
def compute_loss(params, x, y):
    y_pred = network(params, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss

# Generate data under a known model ('target_params=0.5')
key = jax.random.PRNGKey(42)
target_params = 0.5 # [0.5, 0.5]
# Generate some data
xs = jax.random.normal(key, (16, 2))
ys = jnp.sum(xs * target_params, axis=-1)

# Basic usage of Optax
# Optax contains implementations of many popular optimizers that can be used
# very simply. For example, the gradient transform for the `Adam` optimizer is
# available at `optax.adam()`.
# 
# Start to call `GradientTransform` object for Adam (`optimizer`)
# Initialize the optimizer state using the `init` function and `params` of the network
start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)
# Initialize parameters of the model + optimizer
params = jnp.array([0.0, 0.0])
opt_state = optimizer.init(params)

# A simple update loop
for _ in range(1000):
    grads = jax.grad(compute_loss)(params, xs, ys)
    # The `GradientTransform` object(optimizer) contains an `update` function
    # that takes in current optimizer state and gradients and returns the 'update'
    # that need to be applied to the parameters.
    updates, opt_state = optimizer.update(grads, opt_state)
    # Optax comes with a few simple `update rules` that apply the updates from
    # the gradient transforms to the current parameters to return new ones.
    params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params), 'Optimization should retrive target params used to generate data.'



# Custom optimizers
# Optax makes it easy to create custom optimizers by `chain`ing gradient transforms.
## Exponential decay of the learning rate
scheduler = optax.exponential_decay(
    init_value=start_learning_rate,
    transition_steps=1000,
    decay_rate=0.99)

## Combining gradient transforms using 'optax.chain'
gradient_transform = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(), # Use the updates from adam
    optax.scale_by_schedule(scheduler), # Use the learning rate from the scheduler
    # Scale updates by -1 since `optax.apply_updates` is additive and we want to
    # descend on the loss
    optax.scale(-1.0))

# Initialize parameters of the model + optimizer
params = jnp.array([0.0, 0.0])
opt_state = gradient_transform.init(params)
# A simple update loop
for _ in range(1000):
    grads = jax.grad(compute_loss)(params, xs, ys)
    updates, opt_state = gradient_transform.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

print(params)
assert jnp.allclose(params, target_params), 'Optimization should retrive target params used to generate data.'



# Advanced usage of Optax: Modify hyperparameters of optimizers in a schedule.
# In some scenarios, changing the hyperparameters (other than the learning rate)
# of an optimizer can be useful to ensure training reliability. We can do this
# easily by using `optax.inject_hyperparameters`.
decaying_global_norm_tx = optax.inject_hyperparams(optax.clip_by_global_norm)(
    max_norm=optax.linear_schedule(1.0, 0.0, transition_steps=100))

opt_state = decaying_global_norm_tx.init(None)
assert opt_state.hyperparams['max_norm'] == 1.0, 'Max norm should start at 1.0'

for _ in range(100):
    _, opt_state = decaying_global_norm_tx.update(None, opt_state)

assert opt_state.hyperparams['max_norm'] == 0.0, 'Max norm should end at 0.0'