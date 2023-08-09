"""Define Unet

Based on the tutorial of tensorflow.
"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.05'


import tensorflow as tf
from tensorflow.keras import models, layers, losses

import numpy as np
import cv2

"""
Define Layers
"""
class Conv_block(layers.Layer):
    def __init__(self, num_filters): # , name='conv_block'
        super().__init__() #name=name
        self.num_filters = num_filters
        self.conv2d_1 = layers.Conv2D(num_filters, (3, 3), padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.activation_1 = layers.Activation('relu')
        self.conv2d_2 = layers.Conv2D(num_filters, (3, 3), padding='same')
        self.bn_2 = layers.BatchNormalization()
        self.activation_2 = layers.Activation('relu')
        
    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        return x


class Encoder(layers.Layer):
    def __init__(self, num_filters): # , name='encoder'
        super().__init__()
        self.num_filters = num_filters
        self.conv_block = Conv_block(num_filters)
        self.mp2d = layers.MaxPool2D((2, 2), strides=(2, 2))
        
    def call(self, inputs):
        x = self.conv_block(inputs)
        x_pool = self.mp2d(x)
        return x_pool, x


class Decoder(layers.Layer):
    def __init__(self, num_filters): # , name='decoder'
        super().__init__()
        self.num_filters = num_filters
        self.conv2d_tr = layers.Conv2DTranspose(num_filters, (2, 2), 
                                                strides=(2, 2), padding='same')
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.conv_block = Conv_block(num_filters)
        
    def call(self, inputs, concat_tensor):
        x = self.conv2d_tr(inputs)
        x = layers.concatenate([x, concat_tensor], axis=-1)
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv_block(x)
        return x


"""
Define model
"""
class Unet(tf.keras.Model):
    def __init__(self, n_filters_list, n_classes=2, dynamic=True, **kwargs): # , name='u_net'
        assert type(n_filters_list) is list, "n_filters_list must be a list"
        super().__init__(dynamic=dynamic, **kwargs)
        self.conv_block_center = Conv_block(n_filters_list[-1] * 2)
        self.conv_block_final = Conv_block(n_filters_list[0])

        # output layer
        assert n_classes >= 2, "Output classes must >= 2"
        if n_classes == 2:
            self.con2d_1x1 = layers.Conv2D(1, (1, 1), activation='sigmoid')
        else:
            self.con2d_1x1 = layers.Conv2D(n_classes, (1, 1), activation='softmax')
        
        self.encoders = [Encoder(n_filters) for n_filters in n_filters_list]
        self.decoders = [Decoder(n_filters) for n_filters in reversed(n_filters_list)]


    def call(self, inputs):
        # Encoding cascade
        x_pools = []
        xs = []
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                tmp_pool, tmp = encoder(inputs)                
            else:
                tmp_pool, tmp = encoder(x_pools[i-1])
                
            x_pools.append(tmp_pool)
            xs.append(tmp)
        
        # Center convolution block
        z = self.conv_block_center(x_pools[-1])
        
        # Decoding cascade
        xs = xs[::-1] # to match the order of decoders
        for i, decoder in enumerate(self.decoders):
            z = decoder(z, xs[i])
        
        # Final 1x1 convolution
        z = self.conv_block_final(z)
        z = self.con2d_1x1(z)
        return z


"""
Loss functions
"""
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true = tf.reshape(y_true, shape=[-1])
    y_pred = tf.reshape(y_pred, shape=[-1])
    
    intersect = tf.reduce_sum(y_true * y_pred)
    score = (2. * intersect + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return score


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def weighted_cce_loss(y_true, y_pred, **kwargs):
    """y_true is one-hot code"""

    # class balance
    N = tf.reduce_sum(y_true, axis=(1,2))
    W = tf.expand_dims(1 / N, axis=1)
    W = tf.expand_dims(W, axis=2)
    cb = tf.reduce_mean(y_true * W, axis=-1)  # so that sum of weight is 1

    # make max = 1
    cb_max = tf.reduce_max(cb, axis=(1, 2))
    cb_max = tf.expand_dims(tf.expand_dims(cb_max, axis=-1), axis=-1)
    cb = cb / cb_max 

    # distance weight
    nuc_ch = kwargs['nuc_ch']
    cell_ch = kwargs['cell_ch']
    sub_kw = {k: kwargs[k] for k in ['w0', 'sigma']}
    nuc_dw = tf.map_fn(lambda x: distance_weight(x.numpy()[...,nuc_ch], **sub_kw), y_true, tf.float32)
    cell_dw = tf.map_fn(lambda x: distance_weight(x.numpy()[...,[nuc_ch, cell_ch]].sum(axis=-1), 
                                                  **sub_kw), 
                        y_true, tf.float32)

    weight = cb + nuc_dw + cell_dw
    loss = losses.categorical_crossentropy(y_true, y_pred) * weight

    return loss


"""
Pixel weight
"""
def balancing_weight_tf(mask):
    """mask is a tensor"""
    mask = tf.cast(mask, tf.bool)
    n_ones = tf.math.count_nonzero(mask, dtype=tf.int32)
    n_zeros = tf.size(mask, out_type=tf.int32) - n_ones
    x = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_ones, tf.float32)
    y = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_zeros, tf.float32)
    wc = tf.where(mask, x, y) 
    wc = wc / tf.reduce_max(wc)

    return wc


def distance_weight(mask, w0=10, sigma=1, **kwargs):
    """mask is a numpy array"""
    
    # bw2label
    n_objs, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    
    # compute distance to each object for every pixel
    H, W = mask.shape
    D = np.zeros([H, W, n_objs])
    
    for i in range(1, n_objs+1):
        bw = np.uint8(lbl==i)
        D[:,:,i-1] = cv2.distanceTransform(1-bw, cv2.DIST_L2, 3)
        
    D.sort(axis=-1)

    weight = w0 * np.exp(-0.5 * (np.sum(D[:,:,:2], axis=-1)**2) / (sigma**2))
    weight[lbl>0] = 0

    return np.float32(weight)


def get_pixel_weights(mask, **kwargs):
    """mask is a tensor"""
    
    mask = tf.squeeze(mask, axis=-1)
    wc = balancing_weight_tf(mask)
    dw = distance_weight(mask.numpy(), **kwargs)

    return wc + dw


def weighted_bce_loss(y_true, y_pred, **kwargs):
    w = tf.map_fn(lambda x: get_pixel_weights(x, **kwargs), y_true, tf.float32)
    loss = losses.binary_crossentropy(y_true, y_pred) * w

    return loss


def weighted_bce_dice_loss(y_true, y_pred, **kwargs):
    loss = weighted_bce_loss(y_true, y_pred, **kwargs) + dice_loss(y_true, y_pred)

    return loss


# Get model
# def get_model(num_filters_list=[32, 64, 128, 256, 512], compiled=True,
#             optimizer='adam', loss=bce_dice_loss,
#             metrics=[dice_loss]):
#   model = Unet(num_filters_list)
#   if compiled:
#       model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
#   return model
