"""Module for data IO, preprocessing, and augmentation"""

__author__ = 'Chien-Hsiang Hsu'
__create_date__ = '2019.04.05'


import os
import glob
import re
import functools
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_filenames(file_dir, file_type, filter_pattern):
    """ Get file names 
        file_type: e.g. '*.png'
        filter_pattern: regular expression of file name to keep

    Return:
        list of file full paths.
    """
    
    fnames = sorted(glob.glob(os.path.join(file_dir, file_type))) 

    if filter_pattern is not None:
        pattern = re.compile(filter_pattern)
        fnames = [f for f in fnames if pattern.search(os.path.basename(f))]       
    
    return fnames


def get_data_filenames(img_dir, mask_dir, img_file_pattern, mask_file_pattern, match_pattern,
                       test_size=0.2, random_state=None, **kwargs):
    """ Get file names of images and mask. Split into training and validation sets.
        img_file_pattern, mask_file_pattern: e.g. '*.png'
        match_pattern: regular expression of file name to keep

    Return:
        tuple of lists file full paths (x_train, x_val, y_train, y_val)
    """
    # Get image mask file names
    x_train_fnames = get_filenames(img_dir, img_file_pattern, match_pattern)
    y_train_fnames = get_filenames(mask_dir, mask_file_pattern, match_pattern)

    # Split into training and validation
    x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames = \
        train_test_split(x_train_fnames, y_train_fnames, test_size=test_size, random_state=random_state)
    
    return x_train_fnames, x_val_fnames, y_train_fnames, y_val_fnames


# Get image and mask from path name
def _get_image_from_path(img_path, channels=1, dtype='uint8', crop_bd_width=0, 
                         resize=None, scale=1.):
    # Read image
    img = tf.image.decode_png(tf.io.read_file(img_path), channels=channels, dtype=dtype)
    
    # Resize 
    if resize is not None:
        img = tf.image.resize(img, size=resize)

    # Remove bounday 100 pixels since masks touching boundaries were removed
    if crop_bd_width > 0:
        w = crop_bd_width
        img = img[w:-w,w:-w,:]

    # Scale the intensity
    img = tf.cast(img, tf.float32) * scale
    
    return img


"""
Data augmentation
"""
# Flip images
def flip_images(to_flip, img, mask):
    """Flip image and mask horizonally and vertically with prob = 0.5 (separately)"""
    if to_flip:
        flip_prob = tf.random.uniform([2], 0, 1) # [horizontal, vertical]
        
        # flip horizontally
        img, mask = tf.cond(tf.less(flip_prob[0], 0.5), 
                            lambda: (tf.image.flip_left_right(img), 
                                     tf.image.flip_left_right(mask)),
                            lambda: (img, mask))
        # flip vertically
        img, mask = tf.cond(tf.less(flip_prob[1], 0.5), 
                            lambda: (tf.image.flip_up_down(img), 
                                     tf.image.flip_up_down(mask)),
                            lambda: (img, mask))
    return img, mask


# Random crop
def random_crop(img, mask, size=[500, 700]):
    if size is not None:
        assert len(size) == 2, "size must have 2 elments"

        # Combine image and mask then crop
        comb = tf.concat([img, mask], axis=2)
        crop_size = comb.shape.as_list()        
        crop_size[:2] = size
        # size.append(comb.shape[2])
        comb = tf.image.random_crop(comb, size=crop_size)

        # Take out copped image and mask
        img_dim = img.shape[-1]
        img = comb[:,:,:img_dim]
        mask = comb[:,:,img_dim:]
    
    return img, mask


# Assembled augmentation function
def _augment(img, mask, crop_size=None, to_flip=False):
    # Crop and flip
    img, mask = random_crop(img, mask, size=crop_size)
    img, mask = flip_images(to_flip, img, mask)    
    
    return img, mask


"""
Input pipeline
"""
def get_dataset(img_paths, mask_paths, read_img_fn=functools.partial(_get_image_from_path),
                preproc_fn=functools.partial(_augment),
                shuffle=False, repeat=True, batch_size=1, threads=AUTOTUNE):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(read_img_fn, num_parallel_calls=threads)

    if mask_paths is not None:
        mask_dataset = tf.data.Dataset.from_tensor_slices(mask_paths)
        mask_dataset = mask_dataset.map(read_img_fn, num_parallel_calls=threads)
        dataset = tf.data.Dataset.zip((dataset, mask_dataset))
        dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
    
    if shuffle:
        n_samples = len(img_paths)
        dataset = dataset.shuffle(n_samples)

    if repeat:
        dataset = dataset.repeat()
    
    return dataset.batch(batch_size)


"""
For visual inspection
"""
def gray2color(I, clr=(255, 0, 0)):
    """I is uint8."""
    clr = np.array(clr).reshape((1,1,3)) / 255.
    I = I/ 255.
    
    if I.ndim==2:
        I = np.expand_dims(I, axis=2)
    return np.uint8(I * clr * 255)


def overlay_mask(I, M, M_pred, ans_clr=(0, 255, 0), pred_clr=(255, 0, 0)):
    """I, M, M_pred are uint8 numpy arrays
    """
    img_w = 0.8
    ans_w = 1
    pred_w = 1

    if I.shape[-1] == 1:
        I = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)

    M = gray2color(M, clr=ans_clr)
    M_pred = gray2color(M_pred, clr=pred_clr)

    # Z = np.zeros_like(I)
    # Z[...,0] = M_pred # red for prediction
    # Z[...,1] = M # green for ground truth
    
    return cv2.addWeighted(cv2.addWeighted(I, img_w, M, ans_w, 0), 1, M_pred, pred_w, 0)


# def overlay_mask(I, M, M_pred, true_color=(0,255,0), pred_color=(255,0,0)):
#     """I, M, M_pred are uint8 numpy arrays
#     """
#     if I.shape[-1] == 1:
#         I = cv2.cvtColor(I,cv2.COLOR_GRAY2RGB)

#     im_pred, contours_pred, _ = cv2.findContours(M_pred.copy(), 
#                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     Z = np.zeros_like(I)
#     if true_color is None:
#         I1 = np.zeros_like(I)
#     else:
#         im, contours, _ = cv2.findContours(M.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         I1 = cv2.drawContours(Z.copy(), contours, -1, true_color, 1)
        
#     I2 = cv2.drawContours(Z.copy(), contours_pred, -1, pred_color, 1)
    
#     I = np.uint8(np.clip(np.float32(I) + np.float32(I1) + np.float32(I2), 0, 255))
    
#     return I


# """
# Pixel weight
# """
# def balancing_weight_tf(mask):
#     """mask is a tensor"""
#     mask = tf.cast(mask, tf.bool)
#     n_ones = tf.math.count_nonzero(mask, dtype=tf.int32)
#     n_zeros = tf.size(mask, out_type=tf.int32) - n_ones
#     x = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_ones, tf.float32)
#     y = tf.ones_like(mask, dtype=tf.float32) / tf.cast(n_zeros, tf.float32)
#     wc = tf.where(mask, x, y)
#     wc = wc / tf.reduce_min(wc)
    
#     return wc


# def distance_weight(mask, w0=10, sigma=1):
#     """mask is a numpy array"""
    
#     # bw2label
#     n_objs, lbl = cv2.connectedComponents(mask.astype(np.uint8))
    
#     # compute distance to each object for every pixel
#     H, W = mask.shape
#     D = np.zeros([H, W, n_objs])
    
#     for i in range(1, n_objs+1):
#         bw = np.uint8(lbl==i)
#         D[:,:,i-1] = cv2.distanceTransform(1-bw, cv2.DIST_L2, 3)
        
#     D.sort(axis=-1)
#     weight = w0 * np.exp(-0.5 * (np.sum(D[:,:,:2], axis=-1)**2) / (sigma**2))
    
#     return np.float32(weight)


# def get_pixel_weights(mask, **kwargs):
#     """mask is a tensor"""
    
#     mask = tf.squeeze(mask, axis=-1)
#     wc = balancing_weight_tf(mask)
#     # dw = wc
#     dw = tf.numpy_function(lambda x: distance_weight(x, **kwargs), [mask], tf.float32)

#     return tf.expand_dims(wc + dw, axis=-1)


# def concat_weight(img, mask, **kwargs):
#     mask = tf.concat([tf.cast(mask, tf.float32), 
#                       get_pixel_weights(mask, **kwargs)], axis=-1)
#     # mask = tf.map_fn(lambda x: tf.concat([tf.cast(x, tf.float32), 
#     #                                       get_pixel_weights(x, **kwargs)], axis=-1), 
#     #                  mask, dtype=tf.float32)
        
#     return img, mask

