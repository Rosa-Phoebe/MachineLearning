# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 12:02:17 2022

Reference:
    © MIT 6.S191: Introduction to Deep Learning
    http://introtodeeplearning.com
    https://github.com/aamini/introtodeeplearning

@author:
"""

"""
    TensorFlow(TF) is a software library extensively used in machine learning.
    TF uses a high-level API called `Keras` that provide a powerful, intuitive
    framework for building and training deep learning models. Learn:
        Basic of computations in TensorFlow
        Keras API
        TensorFlow imperative execution style: `Eager`
    
    Why is TensorFlow called TensorFlow ?
    
    TensorFlow is called 'TensorFlow' because it handles the flow (node/mathematical
    operation) of Tensors, which are data structures that you can think of as
    multi-dimensional arrays. Tensors are represented as n-dimensional arrays of
    base datatypes such as a string or integer -- they provide a way to generalize
    vectors and matrices to higher dimensions.
    
    The `shape` of a Tensor defines its number of dimensions and the size of each
    dimension. The `rank` of a Tensor provides the number of dimensions(n-dimensions)
    -- you can also think of this as the Tensor's order or degree.
"""

# Latest version of TensorFlow: TensorFlow 2 affords great flexibility and the
# ability to imperatively execute operations just like Python. TensorFlow 2 is
# quite similar to Python in its syntax and imperative execution.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 0-d Tensors: a scalar
sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print("`sport` is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("`number` is a {}-d Tensor".format(tf.rank(number).numpy()))

# Vectors and Lists can be used to create 1-d Tensors
sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))

# Consider creating 2-d(matrices) and higher-rank Tensors.
# Define higher-order Tensors
matrix = tf.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(matrix).numpy() == 2

# Define a 4-d Tensor
# Use `tf.zeros` to initialize a 4-d Tensor of zeros with size 10 x 256 x 256 x 3.
# You can think of this as 10 images where each image is RGB 256 x 256.
images = tf.zeros([10, 256, 256, 3])
assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"

# The `shape` of a Tensor provides the number of elements in each Tensor dimension.
# The `shape` is quite useful. Use slicing to access subtensors within a higher-
# rank Tensor.
row_vector = matrix[1]
column_vector = matrix[:, 2]
scalar = matrix[1, 2]

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))



### Computations on Tensors ###
# A convenient way to think about and visualize computations in TensorFlow is in
# terms of graphs. We can define this graph in terms of Tensors, which hold data
# , and the mathematical operations that act on these Tensors in some order.
#
# Each node in the graph represents an operation that takes some input, does some
# computation, and passes its output to another node.

# Create the nodes in the graph and initialize values
a = tf.constant(15)
b = tf.constant(61)

c1 = tf.add(a, b)
# TensorFlow overrides the "+" operation so that it is able to act on Tensors
c2 = a + b
print(c1)
print(c2)

# Construct a simple computation function
def func(a, b):
    "Define the operation for c, d, e (tf.add, tf.subtract, tf.multiply)"
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e

# Call this function to execute the computation graph give some inputs
a, b = 1.5, 2.5
e_out = func(a, b)
print(e_out)



# Neural networks in TensorFlow. We define neural networks in TensorFlow. TF uses
# a high-level API called `Keras` that provides powerful, intuitive framework
# for building and training deep learning models. Tensors can flow through abstract
# types called `Layer` - the building blocks of neural networks operations and
# are used to update weights, compute losses and define inter-layer connectivity.
#
# Define a network Layer
class OurDenseLayer(tf.keras.layers.Layer):
    # n_output_nodes: number of output nodes
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
        
    # input_shape: shape of the input
    def build(self, input_shape):
        d = int(input_shape[-1])
        # Define and initialize parameters: weight matrix W and bias b
        # Note that parameter initialization is random
        self.W = self.add_weight("weight", shape=[d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape=[1, self.n_output_nodes])
        
    def call(self, x):
        ''' Define the operation for z '''
        z = tf.matmul(x, self.W) + self.b
        
        ''' Define the operation for out'''
        y = tf.sigmoid(z)
        return y
    
# Since layer parameters are initialized randomly, we will set a random seed for
# reproducibility.
tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1, 2))
x_input = tf.constant([[1, 2.]], shape=(1, 2))
y = layer.call(x_input)
print(y.numpy())



# Conveniently, TensorFlow has defined a number of `Layers` that are commonly
# used in neural networks like `Dense`. Instead of using a single `Layer` to
# define simple neural network, use the `Sequential` model from Keras and a
# single `Dense` layer to define network. With `Sequential` API, create neural
# networks by stacking together layers like building blocks.

# Defining a neural network using the Sequential API
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Define the number of outputs
n_output_nodes = 3

# First define the model
model = Sequential()

''' Define a dense (fully connected) layer to compute z '''
# Remember: Dense layers are defined by the parameters W and b!
dense_layer = Dense(n_output_nodes, activation='sigmoid') # Add more comments!!!
# Add dense layer to the model
model.add(dense_layer)

# Test model with example input
x_input = tf.constant([[1, 2.]], shape=(1, 2))
''' Feed input into the model and predict the output '''
model_output = model(x_input).numpy()
print(model_output)



# In addition to defining models using the `Sequential` API, we can also define
# neural networks by directly subclassing the `Model` class, which groups layers
# together to enable model training and inference. The `Model` class captures
# what we refer to as a "model" or as a "network". Using Subclassing, we can
# create a class for our model and then define the forward pass through the
# network using the `call` function.
#
# Subclassing affords the flexibility to define custom layers, custom training
# loops, custom activation functions and custom models.

# Define a model using subclassing



from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubclassModel(tf.keras.Model):
    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        ''' Our model consists of a single Dense layer. Define this layer.'''
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')
    
    # In the call function, we define the Model's forward pass
    def call(self, inputs):
        return self.dense_layer(inputs)
    
#  Test `SubclassModel` using example input
n_output_nodes = 3
model = SubclassModel(n_output_nodes)
x_input = tf.constant([[1, 2.]], shape=(1, 2))
print(model.call(x_input))

# Importantly, `Subclassing` affords us a lot of flexibility to define custom
# models. For example, we can use boolean arguments in the `call` function to
# specify different network behaviors, for example different behaviors during
# training and inference.

## Define a model using subclassing and specifying custom behavior
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class IdentityModel(tf.keras.Model):
    # As before, in __init__, we define the Model's layers
    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        ''' model consists of a single Dense layer. Define this layer '''
        self.dense_layer = Dense(n_output_nodes, activation='sigmoid')
        
    ''' Implement behavior where network outputs the input, unchanged, under
        control of the isidentity argument.
    '''
    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        if isidentity:
            return inputs
        return x

n_output_nodes = 3
model = IdentityModel(n_output_nodes)
x_input = tf.constant([[1, 2.]], shape=(1, 2))
out_activate = model.call(x_input)
out_identity = model.call(x_input, isidentity=True)
print(out_activate.numpy(), out_identity.numpy())



# How to actually implement network training with backpropagation
"""
    Automatic differentiation in TensorFlow
    Automatic differentiation is one of the most important parts of TensorFlow
    and is the backbone of training with backpropagation. Use TF GradientTape
    `tf.GradientTape` to trace operations for computing gradients.
    
    When a forward pass is made through the network, all forward-pass operations
    get recorded to a "tape"; then, to compute the gradient, the tape is played
    backwards. By default, the tape is discarded after it is played backwards;
    this means that a particular `tf.GradientTape` can only compute one gradient,
    and subsequent calls throw a runtime error.
    
    However, we can compute multiple gradients over the same computation by
    creating a `persistent` gradient tape.
"""
# Gradient computation with GradientTape
# y = x^2, x = 3.0
x = tf.Variable(3.0)

# Initiate the gradient tape
with tf.GradientTape() as tape:
    # Define the function
    y = x * x
# Access the gradient - derivation of y with respect to x
dy_dx = tape.gradient(y, x)
assert dy_dx.numpy() == 6.0

"""
    In training neural networks, we use differentiation and stochastic gradient
    descent(SGD) to optimize a loss function. Now that we have a sense of how
    `GradientTape` can be used to compute and access derivatives. 
"""
# Function minimization with automatic differentiation and SGD
# Initialize a random value for our initial x
x = tf.Variable([tf.random.normal([1])])
print("Initializing x = {}".format(x.numpy()))

learning_rate = 1e-2 # learning rate for SGD
history = []
# Define the target value
x_f = 4

# We will run SGD for a number of iterations. At each iteration, we compute the
# loss, compute the derivative of the loss with respect to x and perform the SGD
# update.
for i in range(500):
    with tf.GradientTape() as tape:
        ''' define the loss '''
        loss = (x - x_f) ** 2 # "forward pass": record the current loss on the tape
    
    # loss minimization using gradient tape
    grad = tape.gradient(loss, x) # Compute the derivative of the loss w.r.t. x
    new_x = x - learning_rate * grad # SGD update
    x.assign(new_x) # update the value of x
    history.append(x.numpy()[0])

# Plot the evolution of x as we optimize towards x_f
plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')