# -*- coding: utf-8 -*-
import tensorflow as tf


def content_loss(content_weight, content_current, content_target):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, Tensor with shape [1, height, width, channels]
    - content_target: features of the content image, Tensor with shape [1, height, width, channels]
    
    Returns:
    - scalar content loss
    """
    content_loss = content_weight * 2 * tf.nn.l2_loss(content_current - content_target)
    return content_loss


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: Tensor of shape (1, H, W, C) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: Tensor of shape (C, C) giving the (optionally normalized)
      Gram matrices for the input image.
    """
    shapes = tf.shape(features)
    features_reshape = tf.reshape(features, [shapes[1] * shapes[2], shapes[3]]) # (H * W, C)
    G = tf.matmul(tf.transpose(features_reshape), features_reshape)
    if normalize == True:
        G /= tf.cast(shapes[1] * shapes[2] * shapes[3], tf.float32)
    return G


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    style_loss = 0
    for i in range(len(style_layers)):
        G = gram_matrix(feats[style_layers[i]])
        style_loss += style_weights[i] * 2 * tf.nn.l2_loss(G - style_targets[i])
        
    return style_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: Tensor of shape (1, H, W, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    tv_loss = tv_weight * 2 * (tf.nn.l2_loss(img[:, :, 1:, :] - img[:, :, 0:-1, :]) 
                               + tf.nn.l2_loss(img[:, 1:, :, :] - img[:, 0:-1, :, :]))
    return tv_loss