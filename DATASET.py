import random, os, cv2, itertools

import albumentations as A
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from CONFIG import CONFIG

class DATASET:
    def __init__(self, cfg):
        self.cfg = cfg
        self.CV = self._get_folds()
        if cfg['AUGS']:
            self.ttfs, self.vtfs = self._get_aug_transforms(cfg)
        else:
            self.ttfs, self.vtfs = None, None
           
        if cfg['MODEL']['TYPE'] == 'ACNN':
            if cfg['MODEL']['ACNN']['SIZE_WEIGHTING']:
                assert cfg['MODEL']['inCHANNEL'] == 4, "The model input shape does not match with GT shape!"

            else:
                assert cfg['MODEL']['inCHANNEL'] == 3+len(cfg['MODEL']['ACNN']['RESOLUTIONS']), "The model input shape does not match with GT shape!"
        
    
    def get_CV_info(self):
        return self.CV
    
    def _get_folds(self): #---OK
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
                if len(tmp)==0:
                    break
        
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
        
    def _get_filenames(self, fold_name, resolution, mode):  #---OK
        random.seed(self.cfg['RANDOM_STATE'])
        assert self.cfg['DATASET']['NAME'] in ["Lung","Placenta"], f"{self.cfg['DATASET']['NAME']} is not valid dataset!"
        file_names = []
        
        if mode == 'train':
            file_names_list = self.CV[fold_name]['train_ecnn'] if self.cfg['MODEL']['TYPE'] == "ECNN" else self.CV[fold_name]['train_acnn']
            for name in file_names_list:
                file_names += glob(
                    os.path.join(self.cfg['DATASET']['DIR'],
                                 self.cfg['DATASET']['NAME'],
                                 'RGB-input', 
                                 resolution,
                                 name, 
                                 "*.jpg",
                                )
                )
            random.shuffle(file_names)    
            return file_names
        elif mode == 'val':
            file_names_list = self.CV[fold_name]['train_acnn'] if self.cfg['MODEL']['TYPE'] == "ECNN" else self.CV[fold_name]['test']
            for name in file_names_list:
                file_names += glob(
                    os.path.join(self.cfg['DATASET']['DIR'],
                                 self.cfg['DATASET']['NAME'],
                                 'RGB-input',
                                 resolution,
                                 name, 
                                 "*.jpg",)
                )
            return file_names

        elif mode == 'test':
            for name in self.CV[fold_name]['test']:
                file_names += glob(
                    os.path.join(self.cfg['DATASET']['DIR'],
                                 self.cfg['DATASET']['NAME'],
                                 'RGB-input',
                                 resolution,
                                 name, 
                                 "*.jpg",)
                )

            return file_names
        
    def _get_hm_patch_name(self, name, res, comp, fold_name):
        denum = comp/float(res.replace('x',''))
        x, y = int(name.split('cord')[1])-1, int(name.split('cord')[2])-1
        
        trg_w, trg_h = int(self.cfg['DATASET']['WIDTH']//denum), int(self.cfg['DATASET']['HEIGHT']//denum)
        trg_x, trg_y = int(((y/self.cfg['DATASET']['WIDTH'])%denum)*trg_h), int(((x/self.cfg['DATASET']['HEIGHT'])%denum)*trg_w)
        
        x, y = int(x/self.cfg['DATASET']['WIDTH']/denum)*self.cfg['DATASET']['WIDTH']+1, int(y/self.cfg['DATASET']['HEIGHT']/denum)*self.cfg['DATASET']['WIDTH']+1
        return os.path.join(self.cfg['DATASET']['DIR'],
                            self.cfg['DATASET']['NAME'],
                            'HM-input',
                            fold_name,
                            res,
                            name.split('\\')[-2],
                            f"{self.cfg['DATASET']['ACNN_CLASS']}_cord{x}cord{y}cord.png"), (trg_x, trg_w, trg_y, trg_h)
    
    def _get_acnn_hms(self, name, fold_name, HMs):
        """
        load the hms of different res
        """
        resolutions = [self.cfg['MODEL']['ACNN']['RESOLUTIONS'][index] for index in np.argsort([float(x.replace('x','')) for x in self.cfg['MODEL']['ACNN']['RESOLUTIONS']])[::-1]]
        for index, res in enumerate(resolutions):
            if index == 0:
                HMs[:,:,index] = cv2.imread(
                    os.path.join(self.cfg['DATASET']['DIR'],
                                 self.cfg['DATASET']['NAME'],
                                 'HM-input',
                                 fold_name,
                                 res,
                                 name.split('\\')[-2],
                                 name.split('\\')[-1].replace('.jpg','.png').replace('RGB_',f"{self.cfg['DATASET']['ACNN_CLASS']}_")),
                    cv2.IMREAD_GRAYSCALE)
                comp = float(res.replace('x',''))
                
            else:
                trg_name, (trg_x, trg_w, trg_y, trg_h) = self._get_hm_patch_name(name, res, comp, fold_name)
                HMs[:,:,index] = cv2.resize(cv2.imread(trg_name, cv2.IMREAD_GRAYSCALE)[trg_y:trg_y+trg_h,trg_x:trg_x+trg_w], (self.cfg['DATASET']['WIDTH'], self.cfg['DATASET']['HEIGHT']))

    def _get_img(self, name, fold_name):
        
        if self.cfg['MODEL']['TYPE'] == "ECNN":
            return cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
        
        elif self.cfg['MODEL']['TYPE'] == "ACNN":
            
            img = np.zeros((self.cfg['DATASET']['HEIGHT'],
                            self.cfg['DATASET']['WIDTH'],
                            self.cfg['MODEL']['inCHANNEL']),dtype = np.uint8)
            
            img[:,:,:3] = self._HistEQ(cv2.imread(name)) if self.cfg['DATASET']['HISTEQ'] else cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)
            if self.cfg['MODEL']['ACNN']['SIZE_WEIGHTING']:
                img[:,:,3] = cv2.imread(
                    os.path.join(self.cfg['DATASET']['DIR'],
                                 self.cfg['DATASET']['NAME'],
                                 'HM-input',
                                 fold_name,
                                 'weighted-HMs',
                                 name.split('\\')[-2],
                                 name.split('\\')[-1].replace('.jpg','.png').replace('RGB_',f"{self.cfg['DATASET']['ACNN_CLASSES']}_")),
                                        cv2.IMREAD_GRAYSCALE)
            else:
                self._get_acnn_hms(name, fold_name, img[:,:,3:])
                
            return img
        else:
            raise NameError('Wrong Model Type!')
            
    @staticmethod
    def _HistEQ(image):
        
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        
    def _get_mask(self, name):
        if self.cfg['MODEL']['TYPE'] == 'ECNN':
            if self.cfg['DATASET']['NAME'] == 'Lung':
                mask = np.zeros((self.cfg["DATASET"]["HEIGHT"],self.cfg["DATASET"]["HEIGHT"],5),np.float32)
                mask[:,:,0] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BV_"),cv2.IMREAD_GRAYSCALE)>100)
                mask[:,:,1] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BR_"),cv2.IMREAD_GRAYSCALE)>100)
                mask[:,:,2] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","AVL_"),cv2.IMREAD_GRAYSCALE)>100)
                mask[:,:,3] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","AVLW_"),cv2.IMREAD_GRAYSCALE)>100)
                mask[:,:,4] = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_","BG_"),cv2.IMREAD_GRAYSCALE)>100)

            elif self.cfg['DATASET']['NAME'] == 'Placenta':
                mask = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input"),cv2.IMREAD_GRAYSCALE)>100)

            return mask
        
        elif self.cfg['MODEL']['TYPE'] == 'ACNN':
            if self.cfg['DATASET']['NAME'] == 'Lung':
                mask = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input").replace("RGB_",f"{self.cfg['DATASET']['ACNN_CLASS']}_"),cv2.IMREAD_GRAYSCALE)>100)


            elif self.cfg['DATASET']['NAME'] == 'Placenta':
                mask = np.float32(cv2.imread(name.replace("RGB-input","LABEL-input"),cv2.IMREAD_GRAYSCALE)>100)

            return mask
            

    def _get_datagen(self, file_names, mode=None, fold_name=None):
        
        assert mode in ['train','val','test'], f"{mode} is not a valid mode!"
        if mode == 'train':

            while True:
                X_batch = []
                y_batch = []
                for _ in range(self.cfg['TRAIN']['BATCH_SIZE']):
                    name = next(file_names)
                    
                    image = self._get_img(name, fold_name)
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
                    
                    image = self._get_img(name, fold_name)
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
                batch = []
                batch_name = []

                for _ in range(self.cfg['TEST']['BATCH_SIZE']):
                    name = next(file_names)
                    
                    image = self._get_img(name, fold_name)
                    
                    if self.vtfs:
                        augmentations = self.vtfs(image=image)
                        image = augmentations["image"]
                    
                    batch.append(image)
                    batch_name.append(name)

                yield np.array(batch).astype(np.float32), batch_name  
# ====================================================================================================================
    def get_train_loaders(self, fold_name=None, resolution=None,):
        assert fold_name, "The fold name is not defined!"
        
        file_names = self._get_filenames(
            fold_name=fold_name,
            resolution=resolution if self.cfg['MODEL']['TYPE']!="ACNN" else f"{str(max([float(x.replace('x','')) for x in self.cfg['MODEL']['ACNN']['RESOLUTIONS']])).replace('.0','')}x",
            mode='train',
        )
        
        x = file_names[:round((1-self.cfg['DATASET']['SPLIT'])*len(file_names))]
        y = file_names[round((1-self.cfg['DATASET']['SPLIT'])*len(file_names)):]

        random.shuffle(x)
        random.shuffle(y)

        file_train = itertools.cycle(x)
        file_val = itertools.cycle(y)

        train_gen = self._get_datagen(file_train, 'train', fold_name)
        val_gen = self._get_datagen(file_val, 'val', fold_name)

        return (
            train_gen,
            val_gen,
            int(len(x)//self.cfg['TRAIN']['BATCH_SIZE']),
            int(len(y)//self.cfg['TRAIN']['BATCH_SIZE']),
        )
            
    def get_test_loaders(self, fold_name=None, resolution=None, mode=None):
        assert fold_name, "The fold name is not defined!"
        assert resolution, "The resolution is not defined!"

        file_names = self._get_filenames(
            fold_name=fold_name,
            resolution=resolution if self.cfg['MODEL']['TYPE']!="ACNN" else max([float(x.replace('x','')) for x in self.cfg['MODEL']['ACNN']['RESOLUTIONS']]),
            mode=mode,
        )

        file_test = itertools.cycle(file_names)
        test_gen = self._get_datagen(file_test, 'test')

        return test_gen, int(len(file_names)//self.cfg['TEST']['BATCH_SIZE'])
    
    @staticmethod
    def _get_aug_transforms(cfg): #---OK
        train_augs = []
        test_augs = []
        
        if cfg['AUGS']['RAND_BRIGHT_CONTRAST']:
            train_augs.append(
                A.RandomBrightnessContrast(
                    brightness_limit=cfg['AUGS']['RAND_BRIGHT_CONTRAST']['B_LIM'], 
                    contrast_limit=cfg['AUGS']['RAND_BRIGHT_CONTRAST']['C_LIM'], 
                    brightness_by_max=False ,
                    p=cfg['AUGS']['RAND_BRIGHT_CONTRAST']['PROB'],
                )
            )
            
        if cfg['AUGS']['HISTEQ'] and cfg['MODEL']['TYPE'] != "ACNN":
            train_augs.append(
                A.Equalize(mode='cv', by_channels=False, always_apply=True, p=1.0)
            )
            test_augs.append(
                A.Equalize(mode='cv', by_channels=False, always_apply=True, p=1.0)
            )
        
        train_augs.append(A.Resize(height=cfg['DATASET']['HEIGHT'], width=cfg['DATASET']['WIDTH']))
        test_augs.append(A.Resize(height=cfg['DATASET']['HEIGHT'], width=cfg['DATASET']['WIDTH']))
        
        if cfg['AUGS']['FLIP']:
            train_augs.append(A.Flip(p=cfg['AUGS']['FLIP']))
            
        if cfg['AUGS']['NORMALIZE']:
            train_augs.append(A.Normalize(mean=cfg['AUGS']['NORMALIZE']['MEAN'],std=cfg['AUGS']['NORMALIZE']['STD']))
            test_augs.append(A.Normalize(mean=cfg['AUGS']['NORMALIZE']['MEAN'],std=cfg['AUGS']['NORMALIZE']['STD']))
        else:
            train_augs.append(A.ToFloat())
            test_augs.append(A.ToFloat())
            
        return A.Compose(train_augs), A.Compose(test_augs)
    
    
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
            

