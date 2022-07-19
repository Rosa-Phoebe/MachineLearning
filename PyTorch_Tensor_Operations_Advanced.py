# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:10:44 2022

Reference:
    https://github.com/lucmos/DLAI-s2-2020-tutorials/blob/master/02/02_Tensor_operations.ipynb

@author:
"""

import torch
import numpy as np

from typing import Union
import plotly.express as px

from skimage import io
from skimage.transform import resize


print(torch.__version__)

# Utility print function
def print_arr(*arr: Union[torch.Tensor, np.ndarray], prefix: str = "") -> None:
    """ Pretty print tensors, together with their shape and type
    
    :param arr: one or more tensors
    :param prefix: prefix to use when printing the tensors
    """
    print(
        "\n\n".join(
            f"{prefix}{str(x)} <shape: {x.shape}> <dtype: {x.dtype}>" for x in arr
        )
    )

# Utility function (Run on Colab or Ipython notebook)
# When using plotly, pay attention to that often it does not like PyTorch Tensors
# and it does not give any error, just a empty plot.
def plot_row_images(images: Union[torch.Tensor, np.ndarray]) -> None:
    """
    Plots the images in a subplot with multiple rows. Handles correctly grayscale
    images.
    
    :param images: tensor with shape [number of images, width, height, <colors>]
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    fig = make_subplots(rows=1, cols=images.shape[0], specs=[[{}] * images.shape[0]])
    
    # Convert grayscale image to something that go.Image likes
    if images.dim() == 3:
        images = torch.stack((images, images, images), dim=-1)
    elif (images.dim() == 4 and images.shape[-1] == 1):
        images = torch.cat((images, images, images), dim=-1)
    
    assert images.shape[-1] == 3 or images.shape[-1] == 4
    
    for i in range(images.shape[0]):
        i_image = np.asarray(images[i, ...])
        
        fig.add_trace(
            go.Image(z=i_image, zmin=[0, 0, 0, 0], zmax=[1, 1, 1, 1]),
            row=1,
            col=i + 1
        )
    fig.show()

# Need more understanding???      
def plot_3d_point_cloud(cloud: Union[torch.Tensor, np.ndarray]) -> None:
    """
    Plot a single 3D point cloud
    :param cloud: tensor with shape [number of points, coordinates]
    """
    import pandas as pd
    df = pd.DataFrame(np.asarray(cloud), columns=['x', 'y', 'z'])
    fig = px.scatter_3d(df, x=df.x, y=df.y, z=df.z, color=df.z, opacity=1, range_z=[0, 30])
    fig.update_layout({'scene_aspectmode': 'data', 'scene_camera': dict(
          up=dict(x=0., y=0., z=0.),
          eye=dict(x=0., y=0., z=3.)
    )})
    fig.update_traces(marker=dict(size=2,), selector=dict(mode='markers'))
    _ = fig.show()



# Given two vectors, compute difference between all possible pairs of numbers
# and organize those differences in a matrix
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5])
out = x[:, None] - y[None, :]
print_arr(x, y , out)


# Exercise: Seems amazing.
# Plot Usage
x = torch.zeros(300, 300)
a = 150
b = 150
# Just to visualize the starting point
x[a, b] = 1
plot_row_images(x[None, :])

## L1
rows = torch.arange(x.shape[0])
cols = torch.arange(x.shape[1])
# Manual computation of L1
y = (torch.abs(rows - a)[:, None] + torch.abs(cols - b)[None, :])
px.imshow(y).show()

## Parametric computation of Lp, but breaks with p>=10, Why?
p = 8
y = ((torch.abs(rows - a) ** p)[:, None] + (torch.abs(cols - b) ** p)[None, :]) ** (1/p)
px.imshow(y).show()

## This works even with p=10. Why?
p = 100
y = ((torch.abs(rows.double() - a ) ** p )[:, None] + 
     (torch.abs(cols.double() - b) ** p)[None, :]) ** (1/p)
px.imshow(y).show()

## Explain why
p = 10
print(torch.tensor(10, dtype=torch.int) ** p)
print(torch.tensor(10, dtype=torch.double) ** p)



# Final exercises: Exercise 1
# Create input tensor for the exercise
image1 = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Earth_Eastern_Hemisphere.jpg/260px-Earth_Eastern_Hemisphere.jpg')
image1 = torch.from_numpy(resize(image1, (300, 301), anti_aliasing=True)).float()  # Covert to float type
image1 = image1[..., :3]  # remove alpha channel
# image1.shape => torch.Size([300, 301, 3])

image2 = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg/628px-The_Sun_by_the_Atmospheric_Imaging_Assembly_of_NASA%27s_Solar_Dynamics_Observatory_-_20100819.jpg')
image2 = torch.from_numpy(resize(image2, (300, 301), anti_aliasing=True)).float()
image2 = image2[..., :3]  # remove alpha channel
# image2.shape => torch.Size([300, 301, 3])

image3 = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/1920px-Wikipedia-logo-v2.svg.png')
image3 = torch.from_numpy(resize(image3, (300, 301), anti_aliasing=True)).float()
image3 = image3[..., :3]  # remove alpha channel
# image3.shape => torch.Size([300, 301, 3])

source_images = torch.stack((image1, image2, image3), dim=0)
# source_images => torch.Size([3, 300, 301, 3])

# Plot source images
plot_row_images(source_images)

images = torch.einsum('bwhc -> wbch', source_images) # This is require input

# Grey-fy all images together using `images` tensor
y = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float) # 1-rank tensor
gray_images = torch.einsum('wbch, c -> bwh', (images, y))

# What if you want to transpose the images?
gray_images_tr = torch.einsum('wbch, c -> bhw', (images, y))

# Plot the gray images
plot_row_images(gray_images)
# Plot the gray transposed images
plot_row_images(gray_images_tr)



# Define the linear transformation to swap the blue and red
S = torch.tensor([[0, 0, 1],
                  [0, .5, 0],
                  [1, 0, 0]], dtype=torch.float)
# Apply the linear transformation to the color channel
rb_images = torch.einsum('wbch, dc -> bwhd', (images, S))

plot_row_images(rb_images)



# Exercise 3: => Need to understand again ???
# Define some points in R^2
x = torch.arange(100, dtype=torch.float)
y = x**2
data = torch.stack((x, y), dim=0).t() # => torch.Size([100, 2])

px.scatter(x=data[:, 0].numpy(), y=data[:, 1].numpy())

# Define a matrix that encodes a linear transformation
S = torch.tensor([[-1, 0],
                  [0, 1]], dtype=torch.float)
# Apply the linear transformation: the order is important
new_data = torch.einsum('nk, dk -> nd', (data, S))
# The linear transformation correclty maps the basis vectors!
S @ torch.tensor([[0],
                  [1]], dtype=torch.float)
S @ torch.tensor([[1],
                  [0]], dtype=torch.float)
# Plot the new points
px.scatter(x = new_data[:, 0].numpy(), y = new_data[:, 1].numpy())



## Exercise 4: Convert each image into 3D point cloud
# Just normalize the tensor into the common form [batch, width, height, colors]
imgs = torch.einsum('wbch -> bwhc', images)

# The x, y coordinate of point cloud are all the possible pairs of indices (i, j)
row_indices = torch.arange(imgs.shape[1], dtype=torch.float) # torch.Size([300])
col_indices = torch.arange(imgs.shape[2], dtype=torch.float) # torch.Size([301])
xy = torch.cartesian_prod(row_indices, col_indices) # torch.Size([90300, 2])

# Compute L2 norm for each pixel in each image
# depth = torch.einsum('bwhc, bwhc -> bwh', (imgs, imgs)) ** (1/2)
depth = imgs.norm(p=2, dim=-1) # torch.Size([3, 300, 301]) bwh

# For every pair (i, j), retrieve L2 norm of that pixel
z = depth[:, xy[:, 0].long(), xy[:, 1].long()] * 10 # torch.Size([3, 90300])

# Adjust dimensions, repeat and concatenate accordingly
xy = xy[None, ...] # torch.Size([1, 90300, 2])
xy = xy.repeat(imgs.shape[0], 1, 1) # torch.Size([3, 90300, 2]) 3 images
z = z[..., None] # torch.Size([3, 90300, 1])
clouds = torch.cat((xy, z), dim= 2) # torch.Size([3, 90300, 3])

plot_3d_point_cloud(clouds[0, ...])
plot_3d_point_cloud(clouds[1, ...])
plot_3d_point_cloud(clouds[2, ...])