import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, Reshape, Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, concatenate
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K

import numpy as np

def Conv2d_Block(input_tensor, n_filters, kernel_size, batchnorm=True, regularizer = None):
    # first layer
    x = Conv2D(filters=n_filters, 
               kernel_size=(kernel_size, kernel_size), 
               kernel_initializer="he_normal",
               padding="same", 
               kernel_regularizer=regularizer, 
               bias_regularizer=regularizer)(input_tensor)
    
    if batchnorm:
        x = BatchNormalization(axis=1, momentum=0.99)(x)
    x = Activation("relu")(x)
   
    # second layer
    x = Conv2D(filters=n_filters, 
               kernel_size=(kernel_size, kernel_size), 
               kernel_initializer="he_normal",
               padding="same", 
               kernel_regularizer=regularizer, 
               bias_regularizer=regularizer)(x)
    
    if batchnorm:
        x = BatchNormalization(axis=1, momentum=0.99)(x)
    x = Activation("relu")(x)

    return x

def TransConv2d_Block(input_tensor, n_filters, skip_tensor, drop_rate, batchnorm, regularizer):
    x = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same') (input_tensor)
    x = Concatenate()([x, skip_tensor])
    x = Dropout(drop_rate)(x)
    x = Conv2d_Block(x, n_filters, 3, batchnorm, regularizer)
    return x

def get_unet(input_shape,
             n_classes,
             n_filters, 
             kernel_size,
             batchnorm,
             regularizer,
             classifier):
    # contracting path
    input_img = Input(input_shape)

    c1 = Conv2d_Block(input_img, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2d_Block(p1, n_filters=n_filters*2, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2d_Block(p2, n_filters=n_filters*4, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2d_Block(p3, n_filters=n_filters*8, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2d_Block(p4, n_filters=n_filters*16, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2d_Block(u6, n_filters=n_filters*8, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u7 = Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2d_Block(u7, n_filters=n_filters*4, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u8 = Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2d_Block(u8, n_filters=n_filters*2, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u9 = Conv2DTranspose(n_filters*1, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2d_Block(u9, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    outputs = Conv2D(n_classes, (1, 1), activation= classifier) (c9)
    if classifier == 'sigmoid':
        outputs = Reshape((input_shape[0], input_shape[1]))(outputs)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def get_effnet_unet(input_shape, n_classes=None, drop_rate=0.0, batchnorm=True, regularizer = None):
    inputs = Input(input_shape)
    
    #--- pre-traned ResNet101
    Backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    
    #--- Encoder
    skip1 = Backbone.get_layer("input_1").output                      #--- 480*640  - 4
    skip2 = Backbone.get_layer("block2a_expand_activation").output    #--- 240*320  - 96
    skip3 = Backbone.get_layer("block3a_expand_activation").output    #--- 120*160  - 144
    skip4 = Backbone.get_layer("block4a_expand_activation").output    #--- 60*80    - 240
    skip5 = Backbone.get_layer("block6a_expand_activation").output    #--- 30*40    - 672

    #--- bottleneck
    bridge = Backbone.get_layer("top_activation").output      #--- 15*20 - 1280
    
#     print(skip1.shape,skip2.shape,skip3.shape,skip4.shape,skip5.shape,bridge.shape)
    #--- Decoder
    d1 = TransConv2d_Block(bridge, skip5.shape[-1], skip5, drop_rate, batchnorm, regularizer)       #--- 30*40   
    d2 = TransConv2d_Block(d1, skip4.shape[-1], skip4, drop_rate, batchnorm, regularizer)           #--- 60*80
    d3 = TransConv2d_Block(d2, skip3.shape[-1], skip3, drop_rate, batchnorm, regularizer)           #--- 120*160
    d4 = TransConv2d_Block(d3, skip2.shape[-1], skip2, drop_rate, batchnorm, regularizer)           #--- 240*320
    d5 = TransConv2d_Block(d4, skip1.shape[-1], skip1, drop_rate, batchnorm, regularizer)           #--- 480*640
    
    #--- Output
    if n_classes == 1:
        outputs = Conv2D(n_classes, (1,1), padding='same', activation='sigmoid')(d5)
    else:
        outputs = Conv2D(n_classes, (1,1), padding='same', activation='softmax')(d5)
        
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# def weighted_categorical_crossentropy(weights):
#     """
#     A weighted version of keras.objectives.categorical_crossentropy
    
#     Variables:
#         weights: numpy array of shape (C,) where C is the number of classes
    
#     Usage:
#         weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#         loss = weighted_categorical_crossentropy(weights)
#         model.compile(loss=loss,optimizer='adam')
#     """
    
#     weights = K.variable(weights)
#     def loss(y_true, y_pred):
        
#         y_true = tf.cast(y_true, dtype=tf.float32)
#         # scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # clip to prevent NaN's and Inf's
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         # calc
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss
    
#     return loss

def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights)
    
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weights)
    
    return loss 

def weighted_binary_crossentropy(weight):
    weight = tf.constant(weight)
    
    def loss(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weight)
    
    return loss 
    
# def weighted_binary_crossentropy(weight):
#     """
#     A weighted version of keras.objectives.categorical_crossentropy
    
#     Variables:
#         weights: numpy array of shape (C,) where C is the number of classes
    
#     Usage:
#         weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#         loss = weighted_categorical_crossentropy(weights)
#         model.compile(loss=loss,optimizer='adam')
#     """
    
#     weight = K.variable((1-weight)/weight)
#     def loss(y_true, y_pred):
        
#         y_true = tf.cast(y_true, dtype=tf.float32)

#         # clip to prevent NaN's and Inf's
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         # calc
#         loss = -(y_true * K.log(y_pred)*weight + (1.0 - y_true) * K.log(1.0 - y_pred))
#         loss = K.mean(loss, axis=-1)
#         return loss
    
#     return loss

def dice_coef(y_true, y_pred, smooth=1.):
    yT = K.flatten(y_true)
    yP = K.flatten(y_pred)
    intersection = K.sum(yT * yP)
    return (2. * intersection + smooth)/(K.sum(yT) + K.sum(yP) + smooth)

def IoU(y_true, y_pred):
    yT = K.flatten(y_true)
    yP = K.flatten(y_pred)
    intersection = K.sum(yT * yP)
    union = K.sum(yT) + K.sum(yP) - intersection
    return intersection/union

def weighted_dice_coef(weights):

    def DICE(y_true, y_pred, smooth=1.):
        weighted_DICE = 0
        for index in range(y_pred.shape[-1]):
            yT = K.flatten(y_true[:,:,:,index])
            yP = K.flatten(y_pred[:,:,:,index])
            intersection = K.sum(yT * yP)
            weighted_DICE += (2. * intersection + smooth)*weights[index]/(K.sum(yT) + K.sum(yP) + smooth)
            
        return weighted_DICE/np.sum(weights)
    
    return DICE

def weighted_IoU(weights):

    def IOU(y_true, y_pred):
        weighted_IOU = 0
        for index in range(y_pred.shape[-1]):
            yT = K.flatten(y_true[:,:,:,index])
            yP = K.flatten(y_pred[:,:,:,index])
            intersection = K.sum(yT * yP)
            union = K.sum(yT) + K.sum(yP) - intersection
            weighted_IOU += intersection*weights[index]/union
        return weighted_IOU/np.sum(weights)
    
    return IOU

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

