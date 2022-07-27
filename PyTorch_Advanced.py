# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 19:37:24 2022

Reference:
    https://effectivemachinelearning.com/PyTorch/7._Numerical_stability_in_PyTorch
    https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision

@author:
"""

###############################################################################
"""
    # Numerical stability in PyTorch
    
    When using any numerical computation library such as NumPy or PyTorch, it's
    important to note that writing mathematically correct code doesn't necessarily
    lead to correct results. You also need to make sure that the computations
    are stable.
"""
# Simple example. Mathematically, it's easy to see that `x * y / y = x` for any
# non-zero value of `y`. But let's see if that's always true in practice.
import numpy as np

x = np.float32(1)
y = np.float32(1e-50) # y would be stored as `zero`
z = x * y / y
print(z) # prints `nan`

# The reason for the incorrect result is that `y` is simply too small for `float32`
# type. A similar problem occurs when `y` is too large:
y = np.float32(1e39) # y would be stored as `inf`
z = x * y / y
print(z) # prints `nan`

# The smallest positive value that `float32` type can represent is ``
# and anything below that would be stored as zero. Also, any number beyond
# `` would be stored as `inf`.

# numpy.nextafter(x1, x2):
#     x1: Values to find the next representable value of.
#     x2: The direction where to look for the next representable value of x1.
print(np.nextafter(np.float32(0), np.float32(1))) # 1e-45
print(np.finfo(np.float32).max) # 3.4028235e+38



"""
# To make sure that your computations are stable, you want to avoid values with
# small or very large absolute value. This may sound very obvious, but these
# kind of problems can become extremely hard to debug especially when doing
# gradient descent in PyTorch.

# This is because you not only need to make sure that all the values in the
# forward pass are within the valid range of your data types, but also you need
# to make sure of the same for the backward pass(during gradient computation).
"""
# A real example: compute softmax over a vector of logits.
import torch

def unstable_softmax(logits):
    exp = torch.exp(logits)
    return exp / torch.sum(exp)

print(unstable_softmax(torch.tensor([1000., 0.])).numpy()) # [nan  0.]

# Note that computing the exponential of logits for relatively small numbers
# results to gigantic results are out of `float32` range. The largest valid logit
# for naive softmax implementation is `ln(3.4028235e+38) = 88.72`, anything
# beyond that leads to `nan` outcome.
import torch

def softmax(logits):
    exp = torch.exp(logits - torch.max(logits))
    return exp / torch.sum(exp)

print(softmax(torch.tensor([1000., 0.])).numpy()) # print [1. 0.]

# Consider more complicated case and use softmax function to produce probabilities
# from our logits. We then define our loss function to be the cross entropy
# between our predictions and the labels. Recall that cross entropy for a
# categorical distribution.
#
# Naive implementation of cross entropy.
def unstable_softmax_cross_entropy(labels, logits):
    logits = torch.log(softmax(logits))
    return -torch.sum(labels * logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])
xe = unstable_softmax_cross_entropy(labels, logits)
print(xe.numpy()) # prints inf

# Note in above implementation as the softmax output approaches zero, the log's
# output approaches infinity which causes instability in our computation.
#
def softmax_cross_entropy(labels, logits, dim=-1):
    scaled_logits = logits - torch.max(logits)
    normalized_logits = scaled_logits - torch.logsumexp(scaled_logits, dim) # Why???
    return -torch.sum(labels * normalized_logits)

labels = torch.tensor([0.5, 0.5])
logits = torch.tensor([1000., 0.])
xe = softmax_cross_entropy(labels, logits)
print(xe.numpy()) # prints 500

# Verify the gradients are also computed correctly
logits.requires_grad_(True)
xe = softmax_cross_entropy(labels, logits)
g = torch.autograd.grad(xe, logits)[0]
print(g.numpy()) # prints [ 0.5 -0.5]

### Final
# Extra care must be taken when doing gradient descent to make sure that the
# range of your functions as well as the gradients for each layer are within
# a valid range. Exponential and logarithmic functions when used naively are
# espacially problematic because they can map small numbers to enormous ones
# and the other way around.
###############################################################################



###############################################################################
"""
    Faster training with mixed precision
    
    By default, tensors and model parameters in PyTorch are stored in 32-bit
    floating point precision. Training neural networks using 32-bit floats is
    usually stable and doesn't cause major numerical issues, however neural
    network have been shown to perform quite well in 16-bit and even lower
    precisions. Computation in lower precisions can be significantly faster on
    modern GPUs.
    
    It also has extra benefit of using less memory enabling training larger
    models and/or larger batch sizes which can boost performance further.
    
    The problem though is that training in 16 bits often becomes very unstable
    because the precision is usually not enough to perform some operations like
    accumulations.
    
    PyTorch supports training in mixed precision to help this problem. In a
    nutshell mixed-precision training is done by performing some expensive
    operations(like convolutions and matrix multiplications) in 16-bit by casting
    down the inputs while performing other numerically sensitive operations like
    accumulations in 32-bit. This way we get all the benefits of 16-bit computation
    without its drawbacks.
    
    `Autocast` and `GradScaler` to do "automatic mixed-precision training"
"""

# `Autocast`: Help improve runtime performance by automatically casting down
#             data to 16-bit for some computations.
import torch

# Both `x` and `y` are 32-bit tensors, but `autocast` performs matrix multiplication
# in 16-bit while keeping addition operation in 32-bit. By default, addition of
# two tensors in PyTorch results in a cast to higher precision.
x = torch.rand([32, 32]).cuda()
y = torch.rand([32, 32]).cuda()

with torch.cuda.amp.autocast():
    a = x + y
    b = x @ y
print(a.dtype) # prints torch.float32
print(b.dtype) # prints torch.float16

# Motivation: 16-bit precision may not always be enough for some computations.
# One particular case of interest is representing gradient values, a great
# portion of which are usually small values. Representing them with 16-bit floats
# often leads to buffer underflows(i.e. they'd be represented as zeros). This
# makes training neural networks very unstable. 
#
# `GradScalar` is designed to resovle this issue. `GradScalar` actually monitors
# the gradient values and if it detects overflows it skips updates, scaling down
# the scalar factor according to a configurable schedule. The default schedule
# usually works but you may need to adjust that for your use case.
'''
scaler = torch.cuda.amp.GradScaler()

loss = ...
optimizer = ...  # an instance torch.optim.Optimizer

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
'''
# Sample code: Mixed precision training on synthetic problem of learning to
# generate a checkerboard from image coordinates. Use `Google Colab` GPU and
# Compare `single` and `mixed-precision` performance. In practice with larger
# networks, you may see larger boosts in performance using mixed precision.
import torch
import matplotlib.pyplot as plt
import time

# tensor([0, 1, ..., 511])
x = torch.arange(512) # torch.Size([512])
# tensor([[0, 1, ..., 511]])
x = x.unsqueeze(0) # torch.Size([1, 512])
# tensor([[  0,   1,   2,  ..., 509, 510, 511],
#         [  0,   1,   2,  ..., 509, 510, 511],
#         [  0,   1,   2,  ..., 509, 510, 511],
#         ...,
#         [  0,   1,   2,  ..., 509, 510, 511],
#         [  0,   1,   2,  ..., 509, 510, 511],
#         [  0,   1,   2,  ..., 509, 510, 511]])
x = x.repeat([512, 1]) # torch.Size([512, 512])
#tensor([[0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#        [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#        [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#        ...,
#        [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#        [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#        [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980]])
x = x.div(512)

y = torch.arange(512)
# tensor([[0], [1], ..., [511]])
y = y.unsqueeze(1) # torch.Size([512, 1])
#tensor([[  0,   0,   0,  ...,   0,   0,   0],
#        [  1,   1,   1,  ...,   1,   1,   1],
#        [  2,   2,   2,  ...,   2,   2,   2],
#        ...,
#        [509, 509, 509,  ..., 509, 509, 509],
#        [510, 510, 510,  ..., 510, 510, 510],
#        [511, 511, 511,  ..., 511, 511, 511]])
y = y.repeat([1, 512]) # torch.Size([512, 512])
#tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#        [0.0020, 0.0020, 0.0020,  ..., 0.0020, 0.0020, 0.0020],
#        [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
#        ...,
#        [0.9941, 0.9941, 0.9941,  ..., 0.9941, 0.9941, 0.9941],
#        [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
#        [0.9980, 0.9980, 0.9980,  ..., 0.9980, 0.9980, 0.9980]])
y = y.div(512)

#tensor([[[0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#         [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#         [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#         ...,
#         [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#         [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980],
#         [0.0000, 0.0020, 0.0039,  ..., 0.9941, 0.9961, 0.9980]],
#
#        [[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
#         [0.0020, 0.0020, 0.0020,  ..., 0.0020, 0.0020, 0.0020],
#         [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
#         ...,
#         [0.9941, 0.9941, 0.9941,  ..., 0.9941, 0.9941, 0.9941],
#         [0.9961, 0.9961, 0.9961,  ..., 0.9961, 0.9961, 0.9961],
#         [0.9980, 0.9980, 0.9980,  ..., 0.9980, 0.9980, 0.9980]]])
z = torch.stack([x, y], 0) # torch.Size([2, 512, 512])

def grid(width, height):
    hrange = torch.arange(width).unsqueeze(0).repeat([height, 1]).div(width)
    vrange = torch.arange(height).unsqueeze(1).repeat([1, width]).div(height)
    output = torch.stack([hrange, vrange], 0)
    return output

# More understanding ??????
def checker(width, height, freq):
    hrange = torch.arange(width).reshape([1, width]).mul(freq / width / 2.0).fmod(1.0).gt(0.5)
    vrange = torch.arange(height).reshape([height, 1]).mul(freq / height / 2.0).fmod(1.0).gt(0.5)
    output = hrange.logical_xor(vrange).float()
    return output # torch.Size([512, 512])

# Note the inputs are grid coordinates and the target is a checkerboard
inputs = grid(512, 512).unsqueeze(0).cuda() # torch.Size([1, 2, 512, 512])
targets = checker(512, 512, 8).unsqueeze(0).unsqueeze(1).cuda() # torch.Size([1, 1, 512, 512])

# Plot checker board
plt.subplot(1,2,2)
plt.imshow(targets.squeeze().cpu())
plt.show()

# Define convolutional neural network
class Net(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(2, 256, 1), # in_channels, out_channels, kernel_size
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 1, 1))
        
    @torch.jit.script_method
    def forward(self, x):
        return self.net(x)

# Single precision training
net = Net().cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), 0.001)

start_time = time.time()
for i in range(500):
    opt.zero_grad()
    outputs = net(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    opt.step()

print(loss)
print(time.time() - start_time) # 66.6068012714386

plt.subplot(1,2,1)
plt.imshow(outputs.squeeze().detach().cpu())

plt.subplot(1,2,2)
plt.imshow(targets.squeeze().cpu())
plt.show()


# Mixed precision training
net = Net().cuda()
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), 0.001)

scalar = torch.cuda.amp.GradScaler()
start_time = time.time()

for i in range(500):
    opt.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = net(inputs)
        loss = loss_fn(outputs, targets)
    scalar.scale(loss).backward()
    scalar.step(opt)
    scalar.update()

print(loss)
print(time.time() - start_time) # 32.708489179611206

plt.subplot(1,2,1)
plt.imshow(outputs.squeeze().detach().cpu().float()) # float()

plt.subplot(1,2,2)
plt.imshow(targets.squeeze().cpu().float()) # float()
plt.show()
###############################################################################