#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random, os, cv2, itertools

import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from CONFIG import CONFIG


# In[2]:


class DATASET:
    def __init__(self, cfg):
        self.cfg = cfg
        self.CV = self._get_folds()
        if cfg['TRAIN']['AUG']:
            self.ttfs, self.vtfs = self._get_aug_transforms(cfg)
        else:
            self.ttfs, self.vtfs = None, None
        
    
    def get_CV_info(self):
        return self.CV
    
    def _get_folds(self):
        random.seed(self.cfg['RANDOM_STATE'])
        assert self.cfg['TRAIN']['NUM_CV'] < len(self.cfg['DATASET']['WSI_NAMES']), "Number of folds needs to be smaller than number of sample WSIs"
        
        inds = list(range(len(self.cfg['DATASET']['WSI_NAMES']))) 
        random.shuffle(inds)
        
        folds = []
        tmp = inds.copy()
        while len(tmp)!=0:
            for index in range(self.cfg['TRAIN']['NUM_CV']):
                if len(folds) < 5:
                    folds.append([tmp[0]])
                else:
                    folds[index].append(tmp[0])
                tmp.pop(0)
        
        CV = {}
        for index in range(self.cfg['TRAIN']['NUM_CV']):
            CV[f"CV#{index+1}"] = {}
            CV[f"CV#{index+1}"]['test'] = np.array(self.cfg['DATASET']['WSI_NAMES'])[folds[index]]
            tmp = inds.copy()
            [tmp.remove(val) for val in folds[index]]
            random.seed(int(self.cfg['RANDOM_STATE']*index))
            random.shuffle(tmp)
            CV[f"CV#{index+1}"]['train_ecnn'] = np.array(self.cfg['DATASET']['WSI_NAMES'])[tmp[:len(tmp)//2]]
            CV[f"CV#{index+1}"]['train_acnn'] = np.array(self.cfg['DATASET']['WSI_NAMES'])[tmp[len(tmp)//2:]]
            
        return CV
        
    def _get_filenames(self, fold_name, resolution,mode):  
        random.seed(self.cfg['RANDOM_STATE'])
        assert self.cfg['DATASET']['NAME'] in ["Lung","Placenta"], f"{self.cfg['DATASET']['NAME']} is not valid dataset!"
        file_names = []
        
        if self.cfg['MODEL']['TYPE'] == "ECNN":
            if mode == 'train':
                for name in self.CV[fold_name]['train_ecnn']:
                    file_names += glob(
                        os.path.join(self.cfg['DATASET']['DIR'],
                                     self.cfg['DATASET']['NAME'],
                                     "ECNN Train Dataset",
                                     'RGB-input', 
                                     resolution,
                                     name, 
                                     "*.jpg",
                                    )
                    )
                random.shuffle(file_names)    
                return file_names
            else:
                for name in self.CV[fold_name]['train_acnn']:
                    file_names += glob(
                        os.path.join(self.cfg['DATASET']['DIR'],
                                     self.cfg['DATASET']['NAME'],
                                     "ACNN Train Dataset",
                                     fold_name,
                                     'RGB-input', 
                                     name, 
                                     "*.jpg",
                                    )
                    )
                random.shuffle(file_names)    
                return file_names
        else:
            if mode == 'train':
                for name in self.CV[fold_name]['train_acnn']:
                    file_names += glob(
                        os.path.join(self.cfg['DATASET']['DIR'],
                                     self.cfg['MODEL']['TYPE'],
                                     'RGB', 
                                     fold_name,
                                     name, 
                                     "*.jpg",
                                    )
                    )
                random.shuffle(file_names)    
                return file_names
            else:
                pass
    
    def _get_img(self, name):
        
        if self.cfg['MODEL']['TYPE'] == "ECNN":
            return self._HistEQ(cv2.imread(name)) if self.cfg['DATASET']['HISTEQ'] else cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        
        elif self.cfg['MODEL']['TYPE'] == "ACNN":
            img = np.zeros((self.cfg['DATASET']['HEIGHT'],
                            self.cfg['DATASET']['WIDTH'],
                            self.cfg['DATASET']['inCHANNEL']),dtype = np.uint8)
            
            img[:,:,:3] = self._HistEQ(cv2.imread(name)) if self.cfg['DATASET']['HISTEQ'] else cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
            img[:,:,3] = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        
        else:
            raise NameError('Wrong Model Type!')
            
    @staticmethod
    def _HistEQ(image):
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    def _get_mask(self, name):
        if self.cfg['DATASET']['NAME'] == 'Lung':
            mask = np.zeros((self.cfg["DATASET"]["HEIGHT"],self.cfg["DATASET"]["HEIGHT"],5),np.float32)
            mask[:,:,0] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BV_"),cv2.IMREAD_GRAYSCALE)>100)
            mask[:,:,1] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BR_"),cv2.IMREAD_GRAYSCALE)>100)
            mask[:,:,2] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","AVL_"),cv2.IMREAD_GRAYSCALE)>100)
            mask[:,:,3] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","AVLW_"),cv2.IMREAD_GRAYSCALE)>100)
            mask[:,:,4] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BG_"),cv2.IMREAD_GRAYSCALE)>100)

        elif self.cfg['DATASET']['NAME'] == 'Placenta':
            mask = np.float32(cv2.imread(name.replace("RGB-input","LABE-input"),cv2.IMREAD_GRAYSCALE)>100)

        return mask

    def _get_datagen(self, file_names, mode=None):
        
        assert mode in ['train','val','test'], f"{mode} is not a valid mode!"
        if mode == 'train':

            while True:
                X_batch = []
                y_batch = []
                for _ in range(self.cfg['TRAIN']['BATCH_SIZE']):
                    name = next(file_names)
                    
                    image = self._get_img(name)
                    mask = self._get_mask(name)
                    
                    if self.ttfs:
                        augmentations = self.ttfs(image=image, mask=mask)
                        image = augmentations["image"]
                        mask = augmentations["mask"]
                    
                    X_batch.append(image)
                    y_batch.append(mask)

                yield np.array(X_batch).astype(np.float32), np.array(y_batch).astype(np.float32)
                
        if mode == 'val':

            while True:
                X_batch = []
                y_batch = []
                for _ in range(self.cfg['TRAIN']['BATCH_SIZE']):
                    name = next(file_names)
                    
                    image = self._get_img(name)
                    mask = self._get_mask(name)
                    
                    if self.vtfs:
                        augmentations = self.vtfs(image=image, mask=mask)
                        image = augmentations["image"]
                        mask = augmentations["mask"]
                    
                    X_batch.append(image)
                    y_batch.append(mask)

                yield np.array(X_batch).astype(np.float32), np.array(y_batch).astype(np.float32)       
                
        if mode == 'test': 

            while True:
                X_batch = []

                for _ in range(self.cfg['TEST']['BATCH_SIZE']):
                    name = next(file_name)
                    
                    image = self._get_img(name)
                    
                    if self.vtfs:
                        augmentations = self.vtfs(image=image)
                        image = augmentations["image"]
                    
                    X_batch.append(image)

                yield np.array(X_batch).astype(np.float32)        
# ====================================================================================================================
    def get_train_loaders(self, fold_name=None, resolution=None,):
        assert fold_name, "The fold name is not defined!"
        assert resolution, "The resolution is not defined!"
        
        file_names = self._get_filenames(
            fold_name,
            resolution,
            'train',
        )
        
        x = file_names[:round((1-self.cfg['DATASET']['SPLIT'])*len(file_names))]
        y = file_names[round((1-self.cfg['DATASET']['SPLIT'])*len(file_names)):]

        random.shuffle(x)
        random.shuffle(y)

        file_train = itertools.cycle(x)
        file_val = itertools.cycle(y)

        train_gen = self._get_datagen(file_train, 'train')
        val_gen = self._get_datagen(file_val, 'val')

        return (
            train_gen,
            val_gen,
            int(len(x)//self.cfg['TRAIN']['BATCH_SIZE']),
            int(len(y)//self.cfg['TRAIN']['BATCH_SIZE']),
        )
            
    def get_test_loaders(self, fold_name=None, resolution=None):
        assert fold_name, "The fold name is not defined!"
        assert resolution, "The resolution is not defined!"

        file_names = self._get_filenames(
            fold_name,
            resolution,
            'test',
        )

        random.shuffle(file_names)
        file_test = itertools.cycle(file_names)
        test_gen = self._get_datagen(file_test, 'test')

        return test_gen,int(len(x)//self.cfg['TRAIN']['BATCH_SIZE'])
    
    @staticmethod
    def _get_aug_transforms(cfg):
        train_transform =  A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=(0.8,1.2), contrast_limit=(0.8,1.2), brightness_by_max=False ,p=0.5),
#                 A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),          
                A.Resize(height=cfg['DATASET']['HEIGHT'], width=cfg['DATASET']['WIDTH']),
                A.Flip(p=0.5),
                A.ToFloat(),
            ],
        )
        
        val_transform =  A.Compose(
            [
#                 A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),          
                A.Resize(height=cfg['DATASET']['HEIGHT'], width=cfg['DATASET']['WIDTH']),
                A.ToFloat(),
            ],
        )
        
        return train_transform, val_transform
    
    
if __name__=="__main__":
    cfg = CONFIG().get_config()
    
    cfg['DATASET']['DIR'] = "C:/Sina Pfiles/Thesis/Histopathology Image Segmentation Project"
    cfg['DATASET']['NAME'] = 'Lung'
    cfg['DATASET']['HEIGHT'] = 256
    cfg['DATASET']['WIDTH'] = 256
    cfg['DATASET']['SPLIT'] = 0.2
    cfg['DATASET']['HISTEQ'] = False
    
    cfg['TRAIN']['AUG'] = True
    cfg['TRAIN']['NUM_CV'] = 5
    
    dataset = DATASET(cfg)
    
    (train_loader, val_loader,x,y) = dataset.get_train_loaders(fold_name="CV#1", resolution='2.5x',)
    for X,Y in train_loader:
        plt.imshow(X[0])
        plt.show()
        plt.imshow(Y[0,:,:,:3])
        plt.show()
        print(Y[0,:,:,0].max())
            

