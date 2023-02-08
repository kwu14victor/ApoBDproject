import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import *#CapsuleLayer, PrimaryCap, Length, Mask
import os
from ops import update_routing
from math import floor

def AlexNet(input_shape, n_class, batch_size):
    Input = layers.Input(shape=input_shape, batch_size=None)
    conv1 = layers.Conv2D(filters=96, kernel_size=7, strides=2, padding='same', name='conv1')(Input)
    #x = layers.Lambda(tf.nn.local_response_normalization, name='lrn_1')(x)
    relu1 = layers.ReLU(name='relu1')(conv1)
    mp1 = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name = 'MP1')(relu1)
    conv2 = layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', name='conv2')(mp1)
    #x = layers.Lambda(tf.nn.local_response_normalization, name='lrn_2')(x)
    relu2 = layers.ReLU(name='relu2')(conv2)
    mp2 = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name = 'MP2')(relu2)
    conv3_1 = layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3.1')(mp2)
    conv3_2 = layers.Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu', name='conv3.2')(conv3_1)
    conv4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', name='conv4')(conv3_2)
    mp3 = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name = 'MP3')(conv4)
    F = layers.Flatten()(mp3)
    fc1 = layers.Dense(512)(F)
    dp1 = layers.Dropout(0.5)(fc1)
    relu3 = layers.ReLU()(dp1)
    fc2 = layers.Dense(512)(relu3)
    dp2 = layers.Dropout(0.5)(fc2)
    relu4 = layers.ReLU()(dp2)
    dp3 = layers.Dense(2, name='alexnet', activation = 'softmax')(dp2)
        
    return models.Model([Input], [dp3])
    
    
    

def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    #xx = layers.GaussianNoise(0.05)(x)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    #', kernel_regularizer=regularizers.l2(0.0005)
    
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    #, kernel_regularizer=regularizers.l2(0.0005)
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def DeepCapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='valid', activation='relu', name='conv1')(x)
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H, W, 1, C))(conv1)
    #print(conv1_reshaped.shape)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=2, padding='same', routings=3, name='primarycaps')(conv1_reshaped)
    #ConvCapsuleLayer(conv1, dim_capsule=16, n_channels=8, kernel_size=5, strides=2, padding='valid')
    
    #pc_transposed = tf.transpose(primarycaps, [3, 0, 1, 2, 4])
    #pc_shape = K.shape(pc_transposed)
    #pc_reshaped = K.reshape(pc_transposed, [pc_shape[0] * pc_shape[1], 28, 28, 16])
    #pc_reshaped.set_shape((None, 28, 28, 16))
    
    secondarycaps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=2, padding='same', routings=3, name='secondarycaps')(primarycaps)
    #print(secondarycaps)
    #ConvCapsuleLayer(primarycaps, dim_capsule=16, n_channels=8, kernel_size=5, strides=2, padding='valid')
    # Layer 3: Capsule layer. Routing algorithm works here.
    _, H, W, D, dim = secondarycaps.get_shape()
    sec_cap_reshaped = layers.Reshape((H * W * D, dim))(secondarycaps)
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(sec_cap_reshaped)
    
    #print(digitcaps.shape)
    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    
    #print(out_caps.shape)
    
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
    
    
    decoder_input = tf.reshape(masked_by_y, [-1, 2, 4, 4])
    cube = tf.transpose(decoder_input, [0, 2, 3, 1])
    
    decoder_input2 = tf.reshape(masked, [-1, 2, 4, 4])
    cube2 = tf.transpose(decoder_input2, [0, 2, 3, 1])
    
    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    dmode = 0
    if dmode==0:
        decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
        decoder.add(layers.Dense(1024, activation='relu'))
        decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    elif dmode==1:
        decoder.add(layers.Conv2DTranspose(filters=8, kernel_size=2,strides=1,padding = "same",activation= "relu"))
        decoder.add(layers.Resizing(7, 7))
        #decoder.add(layers.UpSampling2D(size = (7/4, 7/4)))
        decoder.add(layers.Conv2DTranspose(filters=16, kernel_size=3,strides=1,padding = "same",activation= "relu"))
        decoder.add(layers.Resizing(13, 13))
        #decoder.add(layers.UpSampling2D(size = (13/7, 13/7)))
        decoder.add(layers.Conv2DTranspose(filters=16, kernel_size=3,strides=1,padding = "same",activation= "relu"))
        decoder.add(layers.Resizing(26, 26))
        #decoder.add(layers.UpSampling2D(size = (25/13, 25/13)))
        decoder.add(layers.Conv2DTranspose(filters=1, kernel_size=3,strides=1,padding = "same",activation= "relu"))
        decoder.add(layers.Resizing(51, 51))
        #decoder.add(layers.UpSampling2D(size = (51/25, 51/25)))
         
    
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    if dmode==0:
        train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = models.Model(x, [out_caps, decoder(masked)])
        manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
     
    elif dmode==1:  
        # Models for training and evaluation (prediction)
        train_model = models.Model([x, y], [out_caps, decoder(cube)])
        eval_model = models.Model(x, [out_caps, decoder(cube2)])
        #eval_model = models.Model(x, [out_caps, decoder(masked)])
    
        # manipulate model
        #noise = layers.Input(shape=(n_class, 16))
        #noised_digitcaps = layers.Add()([digitcaps, noise])
        #masked_noised_y = Mask()([noised_digitcaps, y])
        decoder_input3 = tf.reshape(masked_noised_y, [-1, 2, 4, 4])
        cube3 = tf.transpose(decoder_input3, [0, 2, 3, 1])
        
        #manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
        manipulate_model = models.Model([x, y, noise], decoder(cube3))
    return train_model, eval_model, manipulate_model