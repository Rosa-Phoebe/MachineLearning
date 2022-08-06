# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 23:19:07 2022

    Modular implementation of Capsule Networks based on TensorFlow

Reference:
    Dynamic Routing Between Capsules
    https://github.com/XifengGuo/CapsNet-Keras/tree/tf2.2
    https://colab.research.google.com/github/ageron/handson-ml/blob/master/extra_capsnets.ipynb
    How to implement CapsNets using TensorFlow (Aurélien Géron)
    
    Run on Google Colab

@author:
"""
import os
# import argparse
import math
import numpy as np
import pandas

from PIL import Image

import tensorflow as tf
from tensorflow.keras import initializers, layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

import tensorflow.keras.backend as K

from matplotlib import pyplot as plt



# Set random seeds so that always produce same output
np.random.seed(42)
tf.random.set_seed(42)



def plot_log(filename, show=True):
    data = pandas.read_csv(filename)
    
    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        # training loss
        if key.find('loss') >= 0 and not key.find('val') >= 0:
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')
    
    fig.add_subplot(212)
    for key in data.keys():
        # acc
        if key.find('acc') >= 0:
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')
    
    if show:
        plt.show()

def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
    elif width is not None and height is None:
        height = int(math.ceil(float(num) / width))
    elif height is not None and width is None:
        width = int(math.ceil(float(num) / height))
    
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)
    
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i * shape[0] : (i+1) * shape[0], j * shape[1] : (j+1) * shape[1]] = img[:, :, 0]
    
    return image

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)

# `x_train` shape: (60000, 28, 28, 1), `x_test` shape: (10000, 28, 28, 1)
# `y_train` shape: (60000, 10),        `y_test` shape: (10000, 10)
# `x_train`, `x_test`, `y_train` and `y_test` type: <class 'numpy.ndarray'>
(x_train, y_train), (x_test, y_test) = load_mnist()

# Look at what these hand-written digit images look like:
n_samples = 5

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = x_train[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary") # black and white picture
    plt.axis("off")
plt.show()
    



    
###############################################
# Key layers for constructing Capsule Networks.
###############################################
"""
    The `squash()` function will squash all vectors in the given array, along the
    given axis(by default, the last axis).
    
    Caution: A nasty bug is waiting to bite you - the derivative of ||s|| is
             undefined when ||s|| = 0, so we can't just use `tf.norm()` or else
             it will blow up during training: if a vector is zero, the gradients
             will be `nan`, so when the optimizer updates the variables, they
             will also become `nan`, and from then on you will be stuck in `nan`
             land.
             The solution is to implement the norm manually by computing the
             square root of the sum of squares plus a tiny epsilon value.
"""
def squash(vectors, axis=-1):
    """
    Non-linear activation used in Capsule. It drives length of large vector to
    near 1 and small vector to 0
    
    Params:
        vectors: some vectors to be squashed, N-dim tensor
        axis: axis to squash
    Return:
        Tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

"""
    Estimated Class Probabilities (Length)
    Note: We could just use `tf.norm()` to compute them, but as we saw when
          discussing the squash function, it would be risky.
"""
class Length(layers.Layer):
    """
    Length of output vector of a capsule represent probability that the entity
    represented by capsule is present in the current input.
    Compute length of vectors. Use this layer as model's output.
    inputs: shape = [None, num_vectors, dim_vector]
    ouputs: shape = [None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        # epsilon = 1e-07, A small constant for numerical stability.
        # It's often added to the denominator to prevent a divide by zero error
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + K.epsilon())
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super(Length, self).get_config()
        return config

"""
    Decoder network: Reconstruction
    During training, instead of sending all the outputs of the capsule network
    to the decoder network, we must send only the output vector of the capsule
    that corresponds to the target digit. All other output vectors must be
    masked out.
    
    At inference time, we must mask all output vectors except for the longest
    one, i.e., that one that corresponds to the predicted digit.
"""
class Mask(layers.Layer):
    """
    Mask Tensor with shape=[None, num_capsule, dim_vector] either by capsule
    with max length or by additional input mask. Except max-length capsule (
    or specified capsule), all vectors are masked to zeros. Then flatten masked
    Tensor.
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            # true label is provided with shape=[None, n_classes] (one-hot)
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            # if no true label, mask by max length of capsules (Prediction)
            # compute length of capsules
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            # generate mask which is one-hot code
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])
        
        # inputs.shape = [None, num_capsule, dim_capsule]
        # mask.shape = [None, num_capsule]
        # masked.shape = [None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked
    
    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])
        
    def get_config(self):
        config = super(Mask, self).get_config()
        return config



"""
    Build a Capsule Network to classify these images. Enjoy the ASCII art!
    Note: for readability, left out two arrows:
            Labels → Mask
            Input Images → Reconstruction Loss
    
    
                             Loss
                              ↑
                    ┌─────────┴─────────┐
     Labels → Margin Loss        Reconstruction Loss
                    ↑                   ↑
                   Length            Decoder
                    ↑                   ↑ 
           Digit Capsules ──── Mask ────┘
           ↖↑↗ ↖↑↗ ↖↑↗
           Primary Capsules
                    ↑
            Regular convolution
                    ↑
               Input Images
"""

# In PrimaryCapsules, we have 32x6x6=1152 capsules(8-D capsules). This layer
# will be composed of 32 maps of 6x6 capsules each, where each capsule will
# output an "8D activation vector". (1152 primary capsules)
# PrimaryCapsules converts 20x20 256 channels into 32x6x6 8-D capsules.
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    Params:
        inputs: 4D tensor, shape=[None, width, height, channels]
        dim_capsule: dim of the output vector of capsule
        n_channels: number of types of capsules
    Return:
        output tensor, shape=[None, num_capsule, dim_capsule]
    """
    # output shape -> (batch_size, 6, 6, 256)
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size,
                           strides=strides, padding=padding, name='primarycap_conv2d')(inputs)
    
    # We reshape the output to get a bunch of 8D vectors representing the outputs of the
    # primary capsules. The output of above `Conv2D` is an array containing 32x8=256 feature
    # maps for each instance, where each feature map is 6x6. So the shape of above ouput is
    # (batch_size, 6, 6, 256). We want to chop the 256 into 32 vectors of 8 dimensional each.
    # We could do this by reshaping to (batch_size, 6, 6, 32, 8). However, since the first
    # capsule layer will be fully connected to the next capsule layer, we can simply flatten
    # the 6x6 grids. This means we just need to reshape to (batch_size, 6x6x32, 8)
    # 
    # `layers.Reshape`: Output shape -> (batch_size,) + target_shape
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    
    # Need squash these vectors
    # `layers.Lambda`: Wraps arbitrary expressions as a Layer object.
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


# For primary capsule, to compute the output of the digit capsules, we must first
# compute the predicted output vectors(one for each primary/digit capsule pair).
# Then we can run the routing by agreement algorithm.
#
# For each capsule i in the first layer, we want to predict the output of every
# capsule j in the second layer. For this, we will need a transformation matrix
# W_ij (one for each pair of capsules (i, j)), then we can compute the predicted
# ouput u_j|i = W_ij*u_i. Since we want to transform an 8D vector into a 16D
# vector, each transformation matrix W_ij must have a shape of (16, 8).
class CapsuleLayer(layers.Layer):
    """
    Capsule layer. It is similar to Dense Layer. Dense Layer has `in_num` inputs,
    each is a scalar, output of the neuron from former layer and it has `out_num`
    output neurons. CapsuleLayer just expand output of neuron from scalar to
    vector. So its input shape=[None, input_num_capsule, input_dim_capsule] and
    output shape=[None, out_num_capsule, out_dim_capsule]. For Dense Layer,
    input_dim_capsule=output_dim_capsule=1.
    
    Params:
        num_capsule: number of capsules in this layer
        dim_capsule: dimension of output vectors of capsules in this layer
        routings: number of iterations for routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        assert len(input_shape) >= 3, "Input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        # Transform matrix, from each input capsule to each output capsule
        # Transform matrix - [10, 1152, 16, 8]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')
        self.built = True
        
    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule] => [None, 1152, 8]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule, 1] => [None, 1, 1152, 8, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        
        # Replicate `num_capsule` dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule, 1]
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1]) # [None, 10, 1152, 8, 1]???
        
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
        # Regard first two dimensions as batch dimension, then
        # matmul(W, x): [..., dim_capsule, input_dim_capsule] x [..., input_dim_capsule, 1] -> [..., dim_capsule, 1]
        # [None, num_capsule, input_num_capsule, dim_capsule, 1] => [None, 10, 1152, 16, 1] => [None, 10, 1152, 16]
        # For each pair of first and second layer capsules(10x1152), we have 16D predicted ouput
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))
        
        ### Begin: Routing algorithm
        ### The prior for coupling coefficient, initialized to zeros
        ### Initialize raw routing weights b_ij = 0
        ### b.shape = [None, self.num_capsule, 1, self.input_num_capsule]
        b = tf.zeros(shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])
        
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, 1, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)
            
            # c.shape=[batch_size, num_capsule, 1, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # matmal: [..., 1, input_num_capsule] x [..., input_num_capsule, dim_capsule] -> [..., 1, dim_capsule]
            # outputs.shape=[None, num_capsule, 1, dim_capsule]
            outputs = squash(tf.matmul(c, inputs_hat)) # [None, 10, 1, 16]
            
            if i < self.routings - 1:
                # outpus.shape=[None, num_capsule, 1, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # matmul: [..., 1, dim_capsule] x [..., input_num_capsule, dim_capsule]^T -> [..., 1, input_num_capsule]
                # b.shape=[batch_size, num_capsule, 1, input_num_capsule]
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)
        ### End: Routing algorithm
        
        return tf.squeeze(outputs) # => [None, 10, 1, 16] => [batch_size, 10, 16]
    
    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])
    
    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
            



def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Simple Capsule Network with 3 layers on MNIST.
    params:
        input_shape: data shape [width, height, channels]
        n_class:     number of classes
        routings:    number of routing iterations
        batch_size:  size of batch
    return:
        Two Keras Models: one used for training and other used for evaluation
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size) # [batch_size, width, height, channels] 28x28
    
    # Note: Since we used a kernel size of 9 and no padding (for some reason, that's what "valid" means),
    #       the image shrunk by 9-1=8 pixels after each convolutional layer(Layer 1 and Layer 2: 28x28
    #       to 20x20, then 20x20 to 12x12), and since we used a strides of 2 in the second convolutional
    #       layer(PrimaryCapsule), the image size was divided by 2. This is how we end up with 6x6 feature
    #       maps.
    
    # Layer 1: A conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x) # 20x20
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid') # 6x6
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    # The digit capsule layer contains 10 capsules(one for each digit) of 16 dimension each 
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
    
    # Layer 4: This is an auxiliary layer to replace each capsule with its length.
    #          Just match the true label's shape.
    out_caps = Length(name='capsnet')(digitcaps)
    
    ### Decoder network
    # Add a decoder network on top of the capsule network. It is a regular 3-layer
    # fully connected neural network which will learn to reconstruct the input
    # images based on the output of the capsule network. This will force the
    # capsule network to preserve all the information required to reconstruct
    # the digits across the whole network.
    #
    # This constraint regularizes the model: it reduces the risk of overfitting
    # the training set and it helps generalize to new digits.
    
    y = layers.Input(shape=(n_class,)) # shape => [None, 10]
    # The true label is used to mask the output of capsule layer. For training
    masked_by_y = Mask()([digitcaps, y])
    # Mask using the capsule with maximal length. For prediction
    masked = Mask()(digitcaps) # shape => [100, 160]
    
    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    # Models for training: Functional APIs
    # TF Keras usage: Manipulate complex graph topologies. Models with multiple
    #                  inputs and outputs
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    
    # Models for evaluation (prediction)
    # TF Keras usage: One input and multiple outputs
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    
    # Manipulate model
    noise = layers.Input(shape=(n_class, 16)) # (None, 10, 16)
    noised_digitcaps = layers.Add()([digitcaps, noise]) # (100, 10, 16)
    masked_noised_y = Mask()([noised_digitcaps, y]) # (100, 10, 16)
    # TF Keras usage: multiple inputs and one output
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    
    return train_model, eval_model, manipulate_model

m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
def margin_loss(y_true, y_pred): # Custom loss function in Keras
    """
    Params:
        y_true: [None, n_class]
        y_pred: [None, num_capsule] num_capsule == n_class, y_pred==||v_k||
        
    Return:
        A scalar loss value
    """
    # Compute loss for each instance and each digit: present error + absent error
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lambda_ * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    
    # Sum digit losses for each instance (L_0 + L_1 + ... + L_9) and compute the
    # mean over all instances. => Final margin loss
    return tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')

def train(model, data, args):
    """
    Training a CapsuleNet
    
    Params:
        model: type - models.Model, the CapsuleNet model
        data: tuple containing traning and testing data, like `((x_train, y_train), (x_test, y_test))`
        args: arguments
    Return:
        The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           monitor='val_capsnet_acc',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    
    # compile the model
    # Reconstruction Loss: It's just squared difference between the input image
    # and the reconstructed image. => "mse"
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr),
                  loss=[margin_loss, 'mse'], # Multiple losses correspond to multiple outputs
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
    
    # Training without data augmentation
    """
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]],
              callbacks=[log, tb, checkpoint, lr_decay])
    """
    
    # Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=0.):
        # shift up to 2 pixel for MNIST
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)
            
    # Training with data augmentation.
    # If shift_fraction=0., no augmentation.
    model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=((x_test, y_test), (y_test, x_test)),
              batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay])
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    
    plot_log(args.save_dir + '/log.csv', show=True)
    
    return model
    
def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    
    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstruced images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()
    
def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    # Choose specific digit to manipulate but with different batch size!!! So
    # we can't execute directly after training.
    # This function batch_size=1, so we should re-load model weights or lead to
    # error. Another method -> choose batch_size = 100 too.
    x_test, y_test = data # x_test shape: (10000, 28, 28, 1), y_test shape: (10000, 10)
    index = np.argmax(y_test, axis=1) == args.digit # (10000,) [False False False ... False  True False]
    number = np.random.randint(low=0, high=sum(index) - 1) # random scalar number
    # x_test[index] => (892, 28, 28, 1)
    # x shape: (28, 28, 1), y shape: (10,)
    x, y = x_test[index][number], y_test[index][number]
    # x.shape: (1, 28, 28, 1), y shape: (1. 10)
    x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)
    
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r # tmp shape: (1, 10, 16)
            x_recon = model.predict([x, y, tmp])  # => model.predict([x, y, tmp], batch_size=100)
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


class Args: # For Google Colab, Training
    save_dir = './result'
    epochs = 50
    batch_size = 100
    lr = 0.001
    lr_decay = 0.9
    lam_recon = 0.392
    routings = 3
    shift_fraction = 0.1
    digit = 5 # Digit to manipulate
    weights = None
    testing = False
    
args = Args()

"""    
class Args: # For Google Colab, Testing
    save_dir = './result'
    epochs = 50
    batch_size = 100
    lr = 0.001
    lr_decay = 0.9
    lam_recon = 0.392
    routings = 3
    shift_fraction = 0.1
    digit = 5 # Digit to manipulate
    weights = "result/trained_model.h5"
    testing = True
    
args = Args()
"""

"""
parser = argparse.ArgumentParser(description="Capsule Network on MNIST")
args = parser.parse_args()
args.save_dir = './result'
args.epochs = 50
args.batch_size = 100
args.lr = 0.001
args.lr_decay = 0.9
args.lam_recon = 0.392
args.routings = 3
args.shift_fraction = 0.1
args.digit = 5 # Digit to manipulate
args.weights = None
args.testing = True
"""


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
# Load data
(x_train, y_train), (x_test, y_test) = load_mnist()
# Define model
# x_train.shape[1:] = (28, 28, 1)
model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                              n_class=len(np.unique(np.argmax(y_train, 1))),
                                              routings=args.routings,
                                              batch_size=args.batch_size)
model.summary()
plot_model(model, to_file=args.save_dir + '/model.png', show_shapes=True, rankdir='TB')

# Train or test
# init the model weights with provided one
if args.weights is not None:
    model.load_weights(args.weights)
    
if not args.testing:
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
else:
    if args.weights is None:
        print('No weights are provided. Will test using random initialized weights.')
    manipulate_latent(model=manipulate_model, data=(x_test, y_test), args=args)
    test(model=eval_model, data=(x_test, y_test), args=args)
