import os, random, cv2, itertools, json

import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow, imread
from tensorflow.keras import regularizers, optimizers
from glob import glob

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import load_model, Model

from DATASET import DATASET
from CONFIG import CONFIG
from UTILS import *

class TRAIN:
    def __init__(self, cfg, resolution, fold_name):
        assert cfg['MODEL']['ARCH'] in ['UNET','EFFUNET'], f"{cfg['MODEL']['ARCH']} is not a supported model architecture!"
        
        self.cfg = cfg
        self.res = resolution
        self.fold = fold_name
        self.model = self._get_model()
#         print(self.model.summary())
        self.loss = self._get_loss()
        self.opt = self._get_optimizer()
        self.metric = self._get_metric()
        
    def run_training(self, save_train_history=True):
        """
        modify the self.cfg['DATASET']['inCHANNEL'] and self.cfg['DATASET']['outCHANNEL']
        making it automated assignment!
        """
        CONFIG.dump(self.cfg)
        self.model.compile(
            optimizer=self.opt,
            loss=self.loss,
            metrics=self.metric,
        )        
        early_stopper = EarlyStopping(patience=self.cfg['TRAIN']['PATIENCE'], 
                                      verbose=self.cfg['TRAIN']['VERBOSE'])
        
        check_pointer = ModelCheckpoint(os.path.join(self.cfg['OUTPUT_DIR'], self._get_model_name()),
                                        monitor=self.cfg['TRAIN']['MONITOR'],
                                        verbose=self.cfg['TRAIN']['VERBOSE'], 
                                        save_best_only=self.cfg['TRAIN']['SAVE_BEST_MODEL'], 
                                       )
        
        dataset = DATASET(self.cfg)
        train_loader, val_loader, TSPE, VSPE = dataset.get_train_loaders(fold_name=self.fold, resolution=self.res,)
        
        History = self.model.fit(train_loader,
                                 steps_per_epoch=TSPE, 
                                 validation_data=val_loader, 
                                 validation_steps=VSPE,
                                 epochs = self.cfg['TRAIN']['MAX_EPOCHS'],
                                 callbacks=[early_stopper, check_pointer], 
                                 class_weight=self.lung_class_weights[cfg['DATASET']['ACNN_CLASS']] if self.cfg['TRAIN']['LOSS'] == "WBCE" else None)
        if save_train_history:
            with open(os.path.join(self.cfg['OUTPUT_DIR'], self._get_model_name().replace(".h5",".json").replace("MODEL_","HISTORY_")),'w') as f:
                json.dump(History.history, f)
            return
        else:
            return History

    
    def _get_model_name(self):
        if self.cfg['MODEL']['TYPE'] == "ECNN":
            return f"UNET_{self.cfg['MODEL']['TYPE']}_{self.cfg['DATASET']['NAME']}_{self.res}_{self.fold}.h5"
        
        elif self.cfg['MODEL']['TYPE'] == "ACNN":
            return f"UNET_{self.cfg['MODEL']['TYPE']}_{self.cfg['DATASET']['NAME']}_{self.cfg['DATASET']['ACNN_CLASS']}_{self.fold}.h5"
    
    def _get_model(self):
        if self.cfg['MODEL']['WEIGHTS']:
            return load_model(os.path.join(path_model,self._get_model_name()),
                              custom_objects={'dice_coef':dice_coef,
                                              'IoU':IoU,
                                              'lr':get_lr_metric})
        else:
            if self.cfg['MODEL']['ARCH'] == 'UNET':
                return get_unet(input_shape=(self.cfg['DATASET']['HEIGHT'],self.cfg['DATASET']['WIDTH'],self.cfg['MODEL']['inCHANNEL']),
                                n_classes=self.cfg['MODEL']['outCHANNEL'],
                                n_filters=32,
                                kernel_size = 3,
                                batchnorm=self.cfg['MODEL']['BN'],
                                regularizer = tf.keras.regularizers.l2(self.cfg['SOLVER']['WEIGHT_DECAY']) if self.cfg['MODEL']['REG'] else None,
                                classifier = 'sigmoid' if self.cfg['MODEL']['outCHANNEL']==1 else 'softmax')
            
            elif self.cfg['MODEL']['ARCH'] == 'EFFUNET':
                return get_effnet_unet(input_shape=(self.cfg['DATASET']['HEIGHT'],self.cfg['DATASET']['WIDTH'],self.cfg['MODEL']['inCHANNEL']),
                                       n_classes=self.cfg['MODEL']['outCHANNEL'],
                                       drop_rate=0.0,
                                       batchnorm=self.cfg['MODEL']['BN'],
                                       regularizer = tf.keras.regularizers.l2(self.cfg['SOLVER']['WEIGHT_DECAY']) if self.cfg['MODEL']['REG'] else None,)
    
    def _get_loss(self):
        assert self.cfg['TRAIN']['LOSS'] in ["BCE","CCE","WBCE","WCCE"], f"{self.cfg['TRAIN']['LOSS']} is not a valid loss function!"
        
        if self.cfg['TRAIN']['LOSS'] == "BCE":
            return 'binary_crossentropy'
        
        elif self.cfg['TRAIN']['LOSS'] == "CCE":
            return 'categorical_crossentropy'
        
        elif self.cfg['TRAIN']['LOSS'] == "WBCE":
            return 'binary_crossentropy'
#             return weighted_binary_crossentropy(self.cfg['TRAIN']['LOSS_WEIGHTS'])
        
        elif self.cfg['TRAIN']['LOSS'] == "WCCE":
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
    
    
    

