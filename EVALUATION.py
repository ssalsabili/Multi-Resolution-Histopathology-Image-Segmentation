import os, cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

from IPython.display import clear_output

from UTILS import *
from CONFIG import CONFIG
from DATASET import DATASET

plt.rcParams["figure.figsize"] = (10,10)

class EVALUATION:
    def __init__(self, train_root_dir, 
                 dataset_name,
                 resolution, fold_name, weights=None):
        assert dataset_name in ["Lung","Placenta"], f"{dataset_name} is not a valid dataset!"
        
        self.cfg = CONFIG(config_from_path=train_root_dir).get_config()
        
        self.cfg['AUGS'] = {}
        self.cfg['AUGS']['RAND_BRIGHT_CONTRAST'] = {}
        self.cfg['AUGS']['RAND_BRIGHT_CONTRAST']['B_LIM'] = (0.8,1.2)
        self.cfg['AUGS']['RAND_BRIGHT_CONTRAST']['C_LIM'] = (0.8,1.2)
        self.cfg['AUGS']['RAND_BRIGHT_CONTRAST']['PROB'] = 0.5
        self.cfg['AUGS']['HISTEQ'] = False
        self.cfg['AUGS']['FLIP'] = 0.5
        self.cfg['AUGS']['NORMALIZE'] = None
#         self.cfg['AUGS']['NORMALIZE']['MEAN'] = (0.485, 0.456, 0.406)
#         self.cfg['AUGS']['NORMALIZE']['STD'] = (0.229, 0.224, 0.225)
        
        self.cfg['DATASET']['OUTPUT_DIR'] = os.path.join(self.cfg['DATASET']['DIR'],dataset_name)
        self.res = resolution
        self.fold = fold_name
        if dataset_name=="Lung":
            self.mask_names = ['BV','BR','AVL','AVLW','BG']
        else:
            self.mask_names = None
        
        if weights is None:
            self.cfg['MODEL']['WEIGHTS'] = os.path.join(train_root_dir,self._get_model_name())
            
        else:
            self.cfg['MODEL']['WEIGHTS'] = weights

        self.model = self._get_model()
        
        
    def _get_model_name(self):
        return f"MODEL_{self.cfg['MODEL']['TYPE']}_{self.cfg['DATASET']['NAME']}_UNET_{self.res}_{self.fold}.h5"
        
    def _get_model(self):
        if self.cfg['TRAIN']['LOSS'] == "WCCE":
            model = load_model(self.cfg['MODEL']['WEIGHTS'], 
                               custom_objects={'DICE':weighted_dice_coef, 
                                               'IOU':weighted_IoU,
                                               'loss': weighted_categorical_crossentropy(self.cfg['TRAIN']['LOSS_WEIGHTS']),
                                               'lr':get_lr_metric})
            
        elif self.cfg['TRAIN']['LOSS'] == "WBCE":
            model = load_model(self.cfg['MODEL']['WEIGHTS'], 
                               custom_objects={'DICE':weighted_dice_coef, 
                                               'IOU':weighted_IoU,
                                               'loss': weighted_binary_crossentropy(self.cfg['TRAIN']['LOSS_WEIGHTS']),
                                               'lr':get_lr_metric})
            
        else:
            model = load_model(self.cfg['MODEL']['WEIGHTS'], 
                               custom_objects={'dice_coef':dice_coef, 
                                               'IoU':IoU,
                                               'lr':get_lr_metric})
        return model
    
    def run_evaluation(self, VISU=False, mode=None):
        
        dataset = DATASET(self.cfg)
        test_loader, TSPE = dataset.get_test_loaders(fold_name=self.fold, resolution=self.res,mode=mode)
                
        counter = 0
        path_root = os.path.join(self.cfg['DATASET']['OUTPUT_DIR'],
                            "HM-input",
                            self.fold,
                            self.res,)
        
        print(f"Saving the predictions in {path_root} ...")
        for batch_idx, (batch_img, batch_name) in enumerate(test_loader):
            if batch_idx == TSPE:
                break
            Xpreds = tf.nn.softmax(self.model.predict(batch_img), axis=-1, name=None).numpy()
            Xpreds[Xpreds != Xpreds.max(axis=-1,keepdims=True)] = 0.0
            Xpreds[Xpreds == Xpreds.max(axis=-1,keepdims=True)] = 255.0
            for idx, img in enumerate(batch_img):
                clear_output(wait=True)
                print(f"{batch_idx}/{TSPE}")
                counter += 1
                
                path = os.path.join(path_root,batch_name[idx].split('\\')[-2])
                if not os.path.isdir(path):
                    os.makedirs(path,exist_ok=True)
                
                if self.mask_names:
                    for ch in range(Xpreds[idx].shape[-1]):
                        tmp_name = batch_name[idx].split('\\')[-1].replace('RGB_','').replace('.jpg','.png')
                        cv2.imwrite(os.path.join(path, f"{self.mask_names[ch]}_{tmp_name}"), np.uint8(Xpreds[idx][:,:,ch]))
                        
                else:
                    cv2.imwrite(os.path.join(path, batch_name[idx].split('\\')[-1].replace('.jpg','.png')), np.uint8(Xpreds[idx]))

                if VISU:
                    plt.imshow(img)
                    print(batch_name[idx],Xpreds[idx].max())
                    plt.show()
                    plt.imshow(Xpreds[idx])
                    plt.show()
        
