#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random, cv2, itertools

import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow, imread
from tensorflow.keras import regularizers, optimizers
from glob import glob

from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, Reshape, Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dropout, concatenate
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model, Model

from tensorflow.keras import backend as K

from DATASET import DATASET
from CONFIG import CONFIG


# In[2]:


class TRAIN:
    def __init__(self, cfg, resolution, fold_name):
        self.cfg = cfg
        self.res = resolution
        self.cv = fold_name
        self.model = self._get_model()
        self.loss = self._get_loss()
        self.opt = self._get_optimizer()
        self.metric = self._get_metric()
        
    def run_training(self):
        self.model.compile(
            optimizer=self.opt,
            loss=self.loss,
            metrics=self.metric,
        )
        
        early_stopper = EarlyStopping(patience=self.cfg['TRAIN']['PATIENCE'], 
                                      verbose=self.cfg['TRAIN']['VERBOSE'])
        
        check_pointer = ModelCheckpoint(os.path.join(self.cfg['OUTPUT_DIR'], self._get_model_name()), 
                                        verbose=self.cfg['TRAIN']['VERBOSE'], 
                                        save_best_only=True, 
                                        monitor=self.cfg['TRAIN']['SAVE_BEST_MODEL'])
        
        dataset = DATASET(self.cfg)
        train_loader, val_loader, TSPE, VSPE = dataset.get_train_loaders(fold_name=self.cv, resolution=self.res,)
        
        History = self.model.fit(train_loader,
                                 steps_per_epoch=TSPE, 
                                 validation_data=val_loader, 
                                 validation_steps=VSPE,
                                 epochs = self.cfg['TRAIN']['MAX_EPOCHS'],
                                 callbacks=[early_stopper, check_pointer])
        
        return History

    
    def _get_model_name(self):
        return f"MODEL_{self.cfg['MODEL']['TYPE']}_{self.cfg['DATASET']['NAME']}_UNET_{self.res}_{self.cv}.h5"
    
    def _get_model(self):
        return get_unet(input_shape=(self.cfg['DATASET']['HEIGHT'],self.cfg['DATASET']['WIDTH'],self.cfg['MODEL']['inCHANNEL']),
                        n_classes=self.cfg['MODEL']['outCHANNEL'],
                        n_filters=32,
                        kernel_size = 3,
                        batchnorm=self.cfg['MODEL']['BN'],
                        regularizer = tf.keras.regularizers.l2(self.cfg['MODEL']['REG']) if self.cfg['MODEL']['REG'] else None,
                        classifier = 'sigmoid' if self.cfg['MODEL']['outCHANNEL']==1 else 'softmax')
    
    def _get_loss(self):
        assert self.cfg['TRAIN']['LOSS'] in ["BCE","CCE","WBCE","WCCE"], f"{self.cfg['TRAIN']['LOSS']} is not a valid loss function!"
        
        if self.cfg['TRAIN']['LOSS'] == "BCE":
            return 'binary_crossentropy'
        
        elif self.cfg['TRAIN']['LOSS'] == "CCE":
            return 'categorical_crossentropy'
        
        elif self.cfg['TRAIN']['LOSS'] == "WBCE":
            return weighted_binary_crossentropy(self.cfg['TRAIN']['LOSS_WEIGHTS'])
        
        else:
            return weighted_categorical_crossentropy(self.cfg['TRAIN']['LOSS_WEIGHTS'])

    def _get_optimizer(self):
        assert self.cfg['SOLVER']['NAME'] in ["Adam", "Nadam", "SGD"], f"{self.cfg['SOLVER']['NAME']} is not a valid optimizer!"
        if self.cfg['SOLVER']['NAME'] == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=self.cfg['SOLVER']['LR'],
                                            amsgrad=self.cfg['SOLVER']['AMSGRAD'],)
        
        elif self.cfg['SOLVER']['NAME'] == "Nadam":
            return tf.keras.optimizers.Nadam(learning_rate=self.cfg['SOLVER']['LR'],)
        
        else:
            return tf.keras.optimizers.SGD(learning_rate=self.cfg['SOLVER']['LR'], 
                                           momentum=self.cfg['SOLVER']['MOMENTUM'],
                                           nesterov=self.cfg['SOLVER']['NESTEROV'])
            
            
    def _get_metric(self):
        if self.cfg['MODEL']['outCHANNEL'] > 1:
            return ['categorical_accuracy',
                    weighted_dice_coef(np.ones((5,))), 
                    weighted_IoU(np.ones((5,))),
                    get_lr_metric(self.opt),
                   ]
        else:
            return [tf.keras.metrics.BinaryAccuracy(name='ACC'),
                    dice_coef, 
                    IoU,
                    get_lr_metric(self.opt),
                   ]
        
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, regularizer = None):
    x = Conv2D(filters=n_filters, 
               kernel_size=(kernel_size, kernel_size), 
               kernel_initializer="he_normal",
               padding="same", 
               kernel_regularizer=regularizer, 
               bias_regularizer=regularizer)(input_tensor)

    if batchnorm:
        x = BatchNormalization(axis=1, momentum=0.99)(x)
    x = Activation("relu")(x)

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

def get_unet(input_shape,
             n_classes,
             n_filters, 
             kernel_size,
             batchnorm,
             regularizer,
             classifier):
    # contracting path
    input_img = Input(input_shape)

    c1 = conv2d_block(input_img, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u7 = Conv2DTranspose(n_filters*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u8 = Conv2DTranspose(n_filters*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    u9 = Conv2DTranspose(n_filters*1, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters, kernel_size=kernel_size, batchnorm=batchnorm, regularizer = regularizer)

    outputs = Conv2D(n_classes, (1, 1), activation= classifier) (c9)
#     outputs = Reshape((input_shape[0], input_shape[1]))(outputs)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
        
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def weighted_binary_crossentropy(weight):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weight = K.variable((1-weight)/weight)
    def loss(y_true, y_pred):
        
        y_true = tf.cast(y_true, dtype=tf.float32)

        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = -(y_true * K.log(y_pred)*weight + (1.0 - y_true) * K.log(1.0 - y_pred))
        loss = K.mean(loss, axis=-1)
        return loss
    
    return loss

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

if __name__=="__main__":
    cfg = CONFIG().get_config()

    cfg['OUTPUT_DIR'] = "C:/Sina Pfiles/Thesis/Histopathology Image Segmentation Project/MODEL/lung/ECNN"

    cfg['DATASET']['DIR'] = "C:/Sina Pfiles/Thesis/Histopathology Image Segmentation Project"
    cfg['DATASET']['NAME'] = 'Lung'
    cfg['DATASET']['HEIGHT'] = 256
    cfg['DATASET']['WIDTH'] = 256
    cfg['DATASET']['SPLIT'] = 0.2
    cfg['DATASET']['HISTEQ'] = False

    cfg['MODEL']['TYPE'] = 'ECNN'
    cfg['MODEL']['inCHANNEL'] = 3
    cfg['MODEL']['outCHANNEL'] = 5

    cfg['SOLVER']['NAME'] = "Nadam"

    cfg['TRAIN']['AUG'] = True
    cfg['TRAIN']['NUM_CV'] = 5
    cfg['TRAIN']['LOSS'] = "CCE"
    cfg['TRAIN']['BATCH_SIZE'] = 10

    train = TRAIN(cfg, "2.5x", "CV#1")
    train.run_training()
    
    
    

