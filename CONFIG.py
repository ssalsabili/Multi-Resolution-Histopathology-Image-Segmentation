class CONFIG:
    def __init__(self, config_from_path=None):
        if config_from_path:
            with open(config_from_path if 'config.txt' in config_from_path else os.path.join(config_from_path,'config.txt'), 'r') as f:
                self.config = json.loads(f.read())
        else:
            self.config = self._get_default_config()
            
    def dump(self, dump_dir = None):
        if dump_dir:
            self.config['OUTPUT_DIR'] = dump_dir
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, 'config.txt'), 'w') as file:
                file.write(json.dumps(self.config))
                    
                    
        elif self.config['OUTPUT_DIR']:
            os.makedirs(self.config['OUTPUT_DIR'], exist_ok=True)
            with open(os.path.join(self.config['OUTPUT_DIR'], 'config.txt'), 'w') as file:
                file.write(json.dumps(self.config))
                    
        else:
            raise NameError('OUTPUT directroy is not defined!')
            
    def get_config(self):
        return self.config
            
    def _get_default_config(self):
        config = {'MODEL':{},'SOLVER':{},'DATASET':{},'TRAIN':{},'TEST':{}}
        
        config['OUTPUT_DIR'] = None
        config['RANDOM_STATE'] = 1
        
        '''
        ###########################################################################
        "MODEL":
        - "TYPE": ['ECNN','ACNN']
        - "WEIGHTS": path to a pre-trained model

            
        ###########################################################################
        '''
        config['MODEL']['NAME'] = None
        
        config['MODEL']['TYPE'] = 'ECNN'
        config['MODEL']['ARCH'] = None
        config['MODEL']['WEIGHTS'] = None
        
        config['MODEL']['inCHANNEL'] = None
        config['MODEL']['outCHANNEL'] = None
        
        config['MODEL']['BN'] = True
        config['MODEL']['REG'] = None
        
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
        config['SOLVER']['NAME'] = 'Adam'
        config['SOLVER']['LR'] = 0.001
        config['SOLVER']['WEIGHT_DECAY'] = 0
        config['SOLVER']['AMSGRAD'] = False
        config['SOLVER']['NESTEROV'] = False
        config['SOLVER']['MOMENTUM'] = 0.5
        
        '''
        ###########################################################################
        "Dataset":
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
        config['DATASET']['DIR'] = None
        config['DATASET']['NAME'] = 'Lung'
        
        if config['DATASET']['NAME'] == 'Lung':
            config['DATASET']['WSI_NAMES'] = ['O2 1','O2 2','O2 3','O2 4','O2 5',
                                              'O2 + LPS 1','O2 + LPS 2','O2 + LPS 3','O2 + LPS 4','O2 + LPS 5',
                                              'RA 1','RA 2','RA 3','RA 4','RA 5',
                                              'RA + LPS 1','RA + LPS 2','RA + LPS 3','RA + LPS 4','RA + LPS 5']
        elif config['DATASET']['NAME'] == 'Placenta':
            config['DATASET']['WSI_NAMES'] = ['4737','4747', '4894', '5160','5538','5854', 
                                              '6235','6805','6848', '7790','7998','8098', ]

        config['DATASET']['HEIGHT'] = None
        config['DATASET']['WIDTH'] = None
        
        config['DATASET']['SPLIT'] = None

        config['DATASET']['HISTEQ'] = False
#         config['DATASET']['NORMALIZE'] = False
        
        '''
        ###########################################################################
        "TRAIN":
          - loss function ==> "LOSS" eg. ["BCE","CCE","WBCE","WCCE"] default = none
          - the weights for different labels distribution ==> list eg. [1,2,3,4,5] default = None
          
          - "NUM_CV": (int) number of cross validation folds
          
          - "SAVE_BEST_MODEL" ==> monitor method for saving the best model! ["loss","acc"] default = None
          - "VERBOSE" ==> (int) visualization method in training [0,1,2]
          - "PATIENCE" ==> (int) number of unchanged epochs before breaking the train loop
          - "MAX_EPOCHS" ==> (int) max number of epoch for training
          - "BATCH_SIZE" ==> (int) number of image patches used in training loops
          - "RESOLUTIONS" ==> list(str()) name of resolutions used in ecnn training
          - perfrom augmentation ==> "AUG" default = False, To be implemented!
            
        ###########################################################################
        '''
        config['TRAIN']['LOSS'] = None
        config['TRAIN']['LOSS_WEIGHTS'] = None
        
        config['TRAIN']['NUM_CV'] = 5
        
        config['TRAIN']['SAVE_BEST_MODEL'] = "loss"
        config['TRAIN']['VERBOSE'] = 1
        config['TRAIN']['PATIENCE'] = 25

        config['TRAIN']['MAX_EPOCHS'] = 50
        config['TRAIN']['BATCH_SIZE'] = 1 
        
        config['TRAIN']['RESOLUTIONS'] = ['2.5x','5x','10x','20x']
        ############################################################
        config['TRAIN']['AUG'] = False
        
        '''
        ###########################################################################
        Test Parameters:
          - number of input bacthes ==> "BATCH_SIZE" default = 1
            
        ###########################################################################
        '''
        config['TEST']['BATCH_SIZE'] = 1

        return config
    
        
if __name__=="__main__":
    cfg = CONFIG().get_config()
            
    print(cfg['MODEL'])

