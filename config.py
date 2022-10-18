import os
import yaml
from yacs.config import CfgNode as CN

class Config(CN):

    @classmethod
    def get_instance(cls, args):
        if not hasattr(cls, 'instance'):
            cls.instance = cls(args)
        cls.instance.update_config(args)
        return cls.instance

    def __init__(self, args):
        super().__init__()
        # GENERAL ------------------------------------------
        # self.SAVE_FREQ = 1
        # self.PRINT_FREQ = 10
        self.SEED = 0
        self.TAG = 'default'
        self.OUTPUT = os.path.join('./results', self.TAG)

        # DATA ------------------------------------------
        self.DATA = CN()
        self.DATA.BATCH_SIZE = 128
        self.DATA.PATH = './data'
        self.DATA.NAME = 'kotour'
        self.DATA.NUM_CLASS = 128

        # MODEL ------------------------------------------
        self.MODEL = CN()
        self.MODEL.SCALE = 'base'
        self.MODEL.HIDDEN_DIM = 768 if self.MODEL.SCALE == 'base' else 1024
        self.MODEL.IMAGE = CN()
        self.MODEL.IMAGE.NAME = 'vit'
        self.MODEL.IMAGE.SIZE = 224
        self.MODEL.IMAGE.PATCH = 16
        # self.MODEL.IMAGE.URL = f'google/vit-{self.MODEL.SCALE}-patch{self.MODEL.IMAGE.PATCH}-{self.MODEL.IMAGE.SIZE}'

        self.MODEL.TEXT = CN()
        self.MODEL.TEXT.NAME = 'kcbert'
        # self.MODEL.IMAGE.URL = f'beomi/kcbert-{self.MODEL.SCALE}'

        # TRAIN ----------------------------------------------
        self.TRAIN = CN()
        self.TRAIN.START_EPOCH = 0
        self.TRAIN.EPOCHS = 300
        self.TRAIN.WARMUP_EPOCHS = 20
        self.TRAIN.WEIGHT_DECAY = 0.05
        self.TRAIN.BASE_LR = 2e-5
        self.TRAIN.WARMUP_LR = 2e-7
        self.TRAIN.MIN_LR = 2e-6
        self.TRAIN.CLIP_GRAD = 5.0

        self.TRAIN.LR_SCHEDULER = CN()
        self.TRAIN.LR_SCHEDULER.NAME = 'cosine'
        self.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
        self.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

        self.TRAIN.OPTIMIZER = CN()
        self.TRAIN.OPTIMIZER.NAME = 'adamw'
        self.TRAIN.OPTIMIZER.EPS = 1e-8
        self.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
        self.TRAIN.OPTIMIZER.MOMENTUM = 0.9


    def _update_config_from_file(self, cfg_file):
        self.merge_from_file(cfg_file)


    def update_config(self, args):
        self.defrost()

        def _check_args(name):
            if hasattr(args, name) and eval(f'args.{name}'):
                return True
            return False
        
        if _check_args('output_path'):
            self.OUTPUT = args.output_path
        if _check_args('tag'):
            self.TAG = args.tag
        if _check_args('seed'):
            self.SEED = args.seed

        if _check_args('batch_size'):
            self.DATA.BATCH_SIZE = args.batch_size
        if _check_args('data_path'):
            self.DATA.PATH = args.data_path
        if _check_args('data_name'):
            self.DATA.NAME = args.data_name
        if _check_args('num_class'):
            self.DATA.NUM_CLASS = args.num_class

        if _check_args('model_cfg'):
            self._update_config_from_file(args.model_config)
        if _check_args('train_cfg'):
            self._update_config_from_file(args.train_config)
        self.freeze()