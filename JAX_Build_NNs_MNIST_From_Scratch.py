# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:30:01 2022

Run on Google Colab

Reference:
    https://github.com/gordicaleksa/get-started-with-JAX

@author:
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax import jit, vmap, pmap, grad, value_and_grad

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


seed = 0
mnist_img_size = (28, 28)



def init_MLP(layer_widths, parent_key, scale=0.01):
    params = []
    keys = jax.random.split(parent_key, num=len(layer_widths) - 1)
    
    for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        weight_key, bias_key = jax.random.split(key)
        params.append(
            [
             scale * jax.random.normal(weight_key, shape=(out_width, in_width)),
             scale * jax.random.normal(bias_key, shape=(out_width,))
            ]
        )
    return params

# Test 
key = jax.random.PRNGKey(seed)
MLP_params = init_MLP([784, 512, 256, 10], key)
print(jax.tree_map(lambda x: x.shape, MLP_params))



def MLP_predict(params, x):
    hidden_layers = params[:-1]
    
    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)
    
    w_last, b_last = params[-1]
    logits = jnp.dot(w_last, activation) + b_last
    
    # Need more understandable
    # log(exp(o1)) - log(sum(exp(o1), exp(o2), ..., exp(o10)))
    # => log(exp(o1)/sum(exp(o1), exp(o2), ..., exp(o10)))
    return logits - logsumexp(logits) # equal log(softmax), numerically stability
    
# Test
dummy_img_flat = np.random.randn(np.prod(mnist_img_size))
print(dummy_img_flat.shape)

prediction = MLP_predict(MLP_params, dummy_img_flat)
print(prediction.shape)

# Batched function
batched_MLP_predict = vmap(MLP_predict, in_axes=(None, 0))

dummy_imgs_flat = np.random.randn(16, np.prod(mnist_img_size))
print(dummy_imgs_flat.shape)
predictions = batched_MLP_predict(MLP_params, dummy_imgs_flat)
print(predictions.shape)



# MLP training on MNIST (Check data type carefully!)
## TODO: init MLP
## TODO: add data loading in PyTorch
## TODO: add training loop, loss fn
batch_size = 128

train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=None)
# Check image type in train_dataset
print(train_dataset[0]) # tuple => (training data, training label)

def custom_transform(x):
    # np.ravel: 28 x 28 -> 786
    return np.ravel(np.array(x, dtype=np.float32))

def custom_collate_fn(batch):
    # print(type(batch)) # <class 'list'>
    # print(type(batch[0])) # <class 'tuple'>
    # print(type(batch[0][0])) # <class 'numpy.ndarray'>
    # print(type(batch[0][1])) # <class 'int'>
    transposed_data = list(zip(*batch))
    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])
    return imgs, labels
    
train_dataset = MNIST(root='train_mnist', train=True, download=True, transform=custom_transform)
print(type(train_dataset[0][0])) # training data type
test_dataset = MNIST(root="test_minist", train=False, download=True, transform=custom_transform)

# If we don't use `collate_fn`, this fn will return Tensor, but JAX need Numpy array
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=None, drop_last=True)
# Need `collate_fn` to convert PyTorch Tensor to Numpy array that JAX need
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

# Test
batch_data = next(iter(train_loader))
imgs = batch_data[0]
lbls = batch_data[1]
print(imgs.shape, imgs[0].dtype, lbls.shape, lbls[0].dtype)

# Optimization - loading the whole dataset into memeory
train_images = jnp.array(train_dataset.data).reshape(len(train_dataset), -1)
train_lbls = jnp.array(train_dataset.targets)

test_images = jnp.array(test_dataset.data).reshape(len(test_dataset), -1)
test_lbls = jnp.array(test_dataset.targets)


num_epochs = 10

def loss_fn(params, imgs, gt_lbls):
    predictions = batched_MLP_predict(params, imgs) # log softmax outputs
    return -jnp.mean(predictions * gt_lbls)

def accuracy(params, dataset_imgs, dataset_lbls):
    pred_classes = jnp.argmax(batched_MLP_predict(params, dataset_imgs), axis=1)
    return jnp.mean(dataset_lbls == pred_classes)

@jit
def update(params, imgs, gt_lbls, lr=0.01):
    loss, grads = value_and_grad(loss_fn)(params, imgs, gt_lbls)
    return loss, jax.tree_multimap(lambda p, g: p - lr*g, params, grads)

# Create a MLP
MLP_params = init_MLP([np.prod(mnist_img_size), 512, 256, len(MNIST.classes)], key)
for epoch in range(num_epochs):
    for cnt, (imgs, lbls) in enumerate(train_loader):
        gt_labels = jax.nn.one_hot(lbls, len(MNIST.classes))
        loss, MLP_params = update(MLP_params, imgs, gt_labels)
        
        if cnt % 50 == 0:
            print(loss)
            
    print(f'Epoch {epoch}, train acc = {accuracy(MLP_params, train_images, train_lbls)} test acc = {accuracy(MLP_params, test_images, test_lbls)}')


# Visualizations
## TODO: visualize MLP weights and add predict
## TODO: visualize embeddings using t-SNE
## TODO: visualize dead neurons
import matplotlib.pyplot as plt

imgs, lbls = next(iter(test_loader))
img = imgs[0].reshape(mnist_img_size)
gt_lbl = lbls[0]
print(img.shape)

pred = jnp.argmax(MLP_predict(MLP_params, np.ravel(img)))
print('pred', pred)
print('gt', gt_lbl)
plt.imshow(img)
plt.show()

# Visualize embeddings using t-SNE
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def fetch_activations(params, x):
    hidden_layers = params[:-1]
    
    activation = x
    for w, b in hidden_layers:
        activation = jax.nn.relu(jnp.dot(w, activation) + b)
        
    return activation

batched_fetch_activations = vmap(fetch_activations, in_axes=(None, 0))
imgs, lbls = next(iter(test_loader))

batch_activations = batched_fetch_activations(MLP_params, imgs)
print(batch_activations.shape) # (128, 256)

t_sne_embeddings = TSNE(n_components=2, verbose=1, random_state=123).fit_transform(batch_activations)
print(t_sne_embeddings.shape) # (128, 2)

df = pd.DataFrame()
df["y"] = lbls
df["tsne-1"] = t_sne_embeddings[:, 0]
df["tsne-2"] = t_sne_embeddings[:, 1]
fig, ax = plt.subplots(1)
sns.scatterplot(x="tsne-1", y="tsne-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                s=120,
                data=df).set(title="MNIST data T-SNE projection")
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)