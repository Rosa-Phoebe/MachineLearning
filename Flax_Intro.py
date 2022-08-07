# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 23:18:13 2022

Reference:
    https://github.com/gordicaleksa/get-started-with-JAX
    
    
    Flax was "designed for flexibility" hence the name:
        Flexibility + JAX -> Flax
    What Flax offers is high performance and flexibility (similarly to JAX)
    
    Run on Google Colab
    !pip install flax

@author:
"""
import numpy as np

import jax
from jax import random, numpy as jnp

# NN lab built on top of JAX developed by Google Research
from flax.core import freeze
from flax import linen as nn
from flax.training import train_state # Useful dataclass to keep train state

# JAX optimizers - A separate lib developed by DeepMind
import optax

# Flax doesn't have its own data loading functions - Use PyTorch dataloaders
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



# Start with simplest model possible: single feed-forward layer
model = nn.Dense(features=5)
# All Flax NN layers inherit from `Module` class
# <class 'flax.linen.module.Module'>
print(nn.Dense.__bases__)


# "Inference" with simple model on two steps: init and apply.
# Step 1: init
seed = 23
key1, key2 = random.split(random.PRNGKey(seed))
# Create a dummy input, a 10-dimensional random vector
x = random.normal(key1, (10,))
# Initialization call - gives us the actual model weights
## Note1: automatic shape inference
## Note2: immutable structure (hence FrozenDict)
## Note3: init_with_output if you care, for whatever reason, about the output here
y, params = model.init_with_output(key2, x) # Care about the output
print(y)
print(jax.tree_map(lambda x: x.shape, params)) # Check output to verify Note1/2/3

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

# This is structure that Flax expects. Just for print.
true_params = freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise
key, x_key, noise_key = random.split(key, num=3)
# xs shape = (150, 2); ys shape = (150, 1)
xs = random.normal(x_key, (n_samples, x_dim))
ys = jnp.dot(xs, W) + b
ys += noise_amplitude * random.normal(noise_key, (n_samples, y_dim))

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
    # SGD (naive implementation) -> more advanced optimizer "optax"
    params = jax.tree_multimap(lambda p, g: p - lr * g, params, grads)

    if epoch % log_period_epoch == 0:
        print(f'epoch {epoch}, loss = {loss}')

print('-' * 50)
print(f'Learned params = {params}')
print(f'Gt params = {true_params}')


params = model.init(key, xs) # Start with fresh params again
# Do above same thing but with dedicated optimizers - DeepMind's optax
opt_sgd = optax.sgd(learning_rate=lr)
opt_state = opt_sgd.init(params) # handle state externally
# print(opt_state) # (EmptyState(), EmptyState()) or compare Adam's and SGD's states

for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params)
    updates, opt_state = opt_sgd.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if epoch % log_period_epoch == 0:
        print(f'epoch {epoch}, loss = {loss}')
        
###############################################################################
###############################################################################
###############################################################################
        
"""
    A fully-fledged CNN on MNIST example in Flax - A simple demo
    
    Define model architecture
"""
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

# Add data loading support in PyTorch (Data Pipeline)
def custom_transform(x):
    # Input: (28, 28) uint8 [0, 255] torch.Tensor => Output: (28, 28, 1) float32 [0, 1] np array
    # x: <class 'PIL.Image.Image'>
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.

def custom_collate_fn(batch):
    # Provide us with batches of numpy arrays and not PyTorch's tensors
    transposed_data = list(zip(*batch))
    
    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])
    
    return imgs, labels

mnist_img_size = (28, 28, 1)
batch_size = 128

# We're working with jax and we're not going to use PyTorch Tensors which means
# we need to convert this into numpy array.
# train_dataset type: <class 'torchvision.datasets.mnist.MNIST'>
# After 'custom_transform' => train_dataset[0] => tuple (image numpy array, image label)
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=custom_transform)
test_dataset = MNIST(root='test_mnist', train=False, download=True, transform=custom_transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

# Optimization - loading the whole dataset into memory (train/test dataset)??????
train_images = jnp.array(train_dataset.data) # (60000, 28, 28)
train_lbls = jnp.array(train_dataset.targets) # (60000,)
# Need to convert shape (10000, 28, 28) -> (10000, 28, 28, 1) for test data
# We don't have to do this for train data because `custom_transform` does it for us
test_images = jnp.array(np.expand_dims(test_dataset.data, axis=3)) # (10000, 28, 28, 1)
test_lbls = jnp.array(test_dataset.targets) # (10000,)
    
# Visualize a single mnist image
imgs, lbls = next(iter(test_loader))
img = imgs[0].reshape(mnist_img_size)[:, :, 0] # (28, 28)
gt_lbl = lbls[0]
print(gt_lbl)
plt.imshow(img)
plt.show()


# Utility function
def compute_metrics(*, logits, gt_labels):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)
    
    metrics = {'loss': loss, 'accuracy': accuracy,}
    return metrics

# Define core training functions
@jax.jit
def train_step(state, imgs, gt_labels):
    def loss_fn(params):
        logits = CNN().apply({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=10)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits
    
    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, gt_labels=gt_labels)
    return state, metrics

def train_one_epoch(state, dataloader, epoch):
    # Train for one epoch on the training set.
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels)
        batch_metrics.append(metrics)
        
    # Aggregate the metrics
    # Pull from the accelerator onto host (CPU)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        # ???
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np

@jax.jit
def eval_step(state, imags, gt_labels):
    logits = CNN().apply({'params': state.params}, imgs)
    return compute_metrics(logits=logits, gt_labels=gt_labels)

def evaluate_model(state, test_imgs, test_lbls):
    # Evaluate on the validation set.
    metrics = eval_step(state, test_imgs, test_lbls)
    # Pull from the accelerator onto host (CPU)
    metrics = jax.device_get(metrics)
    metrics = jax.tree_map(lambda x: x.item(), metrics) # np.ndarray -> scalar
    
    return metrics

def create_train_state(key, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(key, jnp.ones([1, *mnist_img_size]))['params']
    sgd_opt = optax.sgd(learning_rate, momentum)
    # `TrainState` is a simple built-in wrapper class that makes things a bit cleaner
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=sgd_opt)

# Finally define the high-level training/val loops
seed = 0
learning_rate = 0.1
momentum = 0.9
num_epochs = 2
batch_size = 32

state = create_train_state(jax.random.PRNGKey(seed), learning_rate, momentum)

for epoch in range(1, num_epochs + 1):
    state, train_metrics = train_one_epoch(state, train_loader, epoch)
    print(f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}")
    
    #test_metrics = evaluate_model(state, test_images, test_lbls) ???
    #print(f"Test epoch: {epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}") ???