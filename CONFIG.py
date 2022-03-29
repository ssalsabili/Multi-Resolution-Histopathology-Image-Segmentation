import os, json

class CONFIG:
    def __init__(self, dataset_name=None, config_from_path=None):
        self.dataset_name = dataset_name
        if config_from_path:
            with open(config_from_path if 'config.json' in config_from_path else os.path.join(config_from_path,'config.json'), 'r') as f:
                self.cfg = json.loads(f.read())
        else:
            self.cfg = self._get_default_config()
          
    @staticmethod
    def dump(cfg, dump_dir = None):
        if dump_dir:
            cfg['OUTPUT_DIR'] = dump_dir
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, 'config.json'), 'w') as file:
                file.write(json.dumps(cfg))
                
            return cfg
                    
        elif cfg['OUTPUT_DIR']:
            os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)
            with open(os.path.join(cfg['OUTPUT_DIR'], 'config.json'), 'w') as file:
                file.write(json.dumps(cfg))
                    
        else:
            raise NameError('OUTPUT directroy is not defined!')
            
    def get_config(self):
        return self.cfg
            
    def _get_default_config(self):
        cfg = {'MODEL':{},'SOLVER':{},'DATASET':{},'AUGS':{},'TRAIN':{},'TEST':{}}
        
        cfg['OUTPUT_DIR'] = None
        cfg['RANDOM_STATE'] = 1
        
        '''
        ###########################################################################
        "MODEL":
            - 'NAME': (str)
            - "TYPE": (str) ['ECNN','ACNN']
            - 'ACNN':
                - 'RESOLUTIONS': list(str)
                - 'SIZE_WEIGHTING': (bool)
                
            - 'ARCH': (str) ['UNET','EFFUNET']
            - "WEIGHTS": (str) path to a pre-trained model
            
            - 'inCHANNEL': (int)
            - 'outCHANNEL': (int)
            
            - 'BN': (bool)
            - 'REG':

        ###########################################################################
        '''
        cfg['MODEL']['NAME'] = None
        
        cfg['MODEL']['TYPE'] = 'ECNN'
        cfg['MODEL']['ACNN'] = {}
        cfg['MODEL']['ACNN']['RESOLUTIONS'] = ['2.5x','5x','10x','20x']
        cfg['MODEL']['ACNN']['SIZE_WEIGHTING'] = False
        
        cfg['MODEL']['ARCH'] = None
        cfg['MODEL']['WEIGHTS'] = None
        
        cfg['MODEL']['inCHANNEL'] = None
        cfg['MODEL']['outCHANNEL'] = None
        
        cfg['MODEL']['BN'] = True
        cfg['MODEL']['REG'] = None
        
        '''
        ###########################################################################
        Solver Parameters:
          - Optimizer Name ==> "Name" eg. ["Adam", "Nadam", "SGD"] default = "Adam"
          - Learning Rate ==> "LR" default = 0.001
          - Weight Decay ==> "WEIGHT_DECAY" default = 0
          - Use amsgrad in Adam/AdamW ==> "AMSGRAD" default = False
          - Use Nesterov in SGD ==> "NESTEROV" default = False
            
        ###########################################################################
        '''
        cfg['SOLVER']['NAME'] = 'Adam'
        cfg['SOLVER']['LR'] = 0.001
        cfg['SOLVER']['WEIGHT_DECAY'] = 0
        cfg['SOLVER']['AMSGRAD'] = False
        cfg['SOLVER']['NESTEROV'] = False
        cfg['SOLVER']['MOMENTUM'] = 0.5
        
        '''
        ###########################################################################
        "Dataset":
          - "OUTPUT_DIR" ==> where to save the predictions
          - root directory ==> "DIR" default = None
          - "NAME": The name of the dataset ['Placenta','Lung']

            Dataset hierarchy
              Train:        [dataset's name]==>[Training Dataset]==>[RGB-input, DEPTH-input, json, LABEL-input]
              Validation:   [dataset's name]==>[Testing Dataset]==>[TEST - annos]==>[RGB-input, DEPTH-input, json, LABEL-input]
              Test:         [dataset's name]==>[Testing Dataset]==>[TEST - no-annos]==>[RGB-input, DEPTH-input]
          
          - image type ==> "TYPE" eg. ["RGB", "RGBD", "G", "GD"] default = "RGB"
          - image height ==> "HEIGHT" default = None
          - image width ==> "WIDTH" default = None
          - data loader use shuffle in training ==> "DL_SHUFFLE" default = True
          - data loader pin memory ==> "DL_PIN_MEMORY" default = True
          - data loader number of workers ==> "DL_NUM_WORKERS" default = 4
          - train-val dataset split ==> "SPLIT" default = None #--- if val directory is not None the split will be ignored
          - number of classes ==> "NUM_CLASSES" default = None
          - label names ==> "CLASS_LABELS" default = None
          - use histogram eq ==> "HISTEQ" default = True
          
            
        ###########################################################################
        '''
        cfg['DATASET']['DIR'] = None
        cfg['DATASET']['OUTPUT_DIR'] = None
        cfg['DATASET']['NAME'] = self.dataset_name
        try:
            if cfg['DATASET']['NAME'] == 'Lung':
                cfg['DATASET']['WSI_NAMES'] = ['O2 1','O2 2','O2 3','O2 4','O2 5',
                                                  'O2 + LPS 1','O2 + LPS 2','O2 + LPS 3','O2 + LPS 4','O2 + LPS 5',
                                                  'RA 1','RA 2','RA 3','RA 4','RA 5',
                                                  'RA + LPS 1','RA + LPS 2','RA + LPS 3','RA + LPS 4','RA + LPS 5']
                cfg['DATASET']['ACNN_CLASS'] = None
            elif cfg['DATASET']['NAME'] == 'Placenta':
                cfg['DATASET']['WSI_NAMES'] = ['4737','4747', '4894', '5160','5538','5854', 
                                                  '6235','6805','6848', '7790','7998','8098', ]
                cfg['DATASET']['ACNN_CLASS'] = ['VILLI']
    
            else:
                cfg['DATASET']['WSI_NAMES'] = None
                print(f"Wrning! The {self.dataset_name} is not valid dataset! Please choose among 'Lung', 'Placenta'")
        except:
            cfg['DATASET']['WSI_NAMES'] = None
            print("Warning! Dataset name is not defined! This is needed for generating train-val folds!")

        cfg['DATASET']['HEIGHT'] = None
        cfg['DATASET']['WIDTH'] = None
        
        cfg['DATASET']['SPLIT'] = None
        
        """
        ###########################################################################
        "AUGS":
            - "RandomBrightnessContrast"
            - "HISTEQ"
            - "FLIP"
            - "NORMALIZE"
        ###########################################################################
        """
        cfg['AUGS']['RAND_BRIGHT_CONTRAST'] = {}
        cfg['AUGS']['RAND_BRIGHT_CONTRAST']['B_LIM'] = (0.8,1.2)
        cfg['AUGS']['RAND_BRIGHT_CONTRAST']['C_LIM'] = (0.8,1.2)
        cfg['AUGS']['RAND_BRIGHT_CONTRAST']['PROB'] = 0.5
        
        cfg['AUGS']['HISTEQ'] = False
    
        cfg['AUGS']['FLIP'] = 0.5
        
        cfg['AUGS']['NORMALIZE'] = {}
        cfg['AUGS']['NORMALIZE']['MEAN'] = (0.485, 0.456, 0.406)
        cfg['AUGS']['NORMALIZE']['STD'] = (0.229, 0.224, 0.225)
                  
        '''
        ###########################################################################
        "TRAIN":
          - loss function ==> "LOSS" eg. ["BCE","CCE","WBCE","WCCE"] default = none
          - the weights for different labels distribution ==> list eg. [1,2,3,4,5] default = None
          
          - "NUM_CV": (int) number of cross validation folds
          
          - "SAVE_BEST_MODEL" ==> ["True","False"] 
          - "MONITOR" ==> ["val_loss","val_acc","val_IoU","val_dice_coef"]
          - "VERBOSE" ==> (int) visualization method in training [0,1,2]
          - "PATIENCE" ==> (int) number of unchanged epochs before breaking the train loop
          - "MAX_EPOCHS" ==> (int) max number of epoch for training
          - "BATCH_SIZE" ==> (int) number of image patches used in training loops
          - "RESOLUTIONS" ==> list(str()) name of resolutions used in ecnn training
          - perfrom augmentation ==> "AUG" default = False, To be implemented!
            
        ###########################################################################
        '''
        cfg['TRAIN']['LOSS'] = None
        cfg['TRAIN']['LOSS_WEIGHTS'] = None
        
        cfg['TRAIN']['NUM_CV'] = 5
        
        cfg['TRAIN']['SAVE_BEST_MODEL'] = True
        cfg['TRAIN']['MONITOR'] = "val_loss"
        cfg['TRAIN']['VERBOSE'] = 1
        cfg['TRAIN']['PATIENCE'] = 25

        cfg['TRAIN']['MAX_EPOCHS'] = 50
        cfg['TRAIN']['BATCH_SIZE'] = 1 
        
        cfg['TRAIN']['RESOLUTIONS'] = ['2.5x','5x','10x','20x']
        ############################################################
        
        '''
        ###########################################################################
        Test Parameters:
          - number of input bacthes ==> "BATCH_SIZE" default = 1
            
        ###########################################################################
        '''
        cfg['TEST']['BATCH_SIZE'] = 1

        return cfg
    
        
if __name__=="__main__":
    cfg = config().get_config()
            
    print(cfg['MODEL'])

