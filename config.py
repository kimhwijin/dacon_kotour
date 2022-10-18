import os
from typing import Text
import yaml
from yacs.config import CfgNode as CN
from functools import partial

def _check_args(name, args):
    if hasattr(args, name) and eval(f'args.{name}'):
        return True
    else:
        return False

class Config(CN):
    def __init__(self, args):
        super().__init__()
        self._check_args = partial(_check_args, args=args)
        self._get_base_config(args)
        self._get_data_config(args)
        self._get_model_config(args)
        self._get_train_config(args)
        
        self.update_config(args)
    
    def _get_base_config(self, args):
        self.SEED = 0                                     if not self._check_args('seed') else args.seed
        self.TAG = 'default'                              if not self._check_args('tag') else args.tag
        self.OUTPUT = os.path.join('./results', self.TAG) if not self._check_args('output') else args.output

    def _get_data_config(self, args):
        self.DATA = CN()
        self.DATA.BATCH_SIZE = 128      if not self._check_args('batch_size') else args.batch_size
        self.DATA.PATH = './data'       if not self._check_args('data_path') else args.data_path
        self.DATA.NAME = 'kotour'       if not self._check_args('data_name') else args.data_name
        self.DATA.NUM_CLASS = 128       if not self._check_args('num_class') else args.num_class

    def _get_model_config(self, args):
        self.MODEL = ModelConfig(args)

    def _get_train_config(self, args):
        self.TRIAN = TrainConfig(args)

    def _update_config_from_file(self, cfg_file):
        self.merge_from_file(cfg_file)

    def update_config(self, args):
        self.defrost()
        
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


class ModelConfig(CN):
    def __init__(self, args):
        super().__init__()
        self._get_base_config()
        self._get_img_model_config(args)
        self._get_txt_model_config(args)
        self._get_config_from_file(args)

    def _get_base_config(self):
        self.SCALE = 'base'
        self.HIDDEN_DIM = 768
    
    def _get_img_model_config(self, args):
        self.IMAGE = ImageModelConfig(args)

    def _get_txt_model_config(self, args):
        self.TEXT = TextModelConfig(args)
    
    def _get_config_from_file(self, args):
        self.merge_from_file(args.model_cfg)

class ImageModelConfig(CN):
    def __init__(self, args):
        super().__init__()
        self._get_config(args)

    def _get_config(self, args):
        if args.img_model == 'vit':
            self._get_vit_config(args)

    def _get_vit_config(self):
        self.NAME = 'vit'
        self.SIZE = 224
        self.PATCH = 16
        self.URL = 'google/vit-base-patch16-224'


class TextModelConfig(CN):
    def __init__(self, args):
        super().__init__()
        self._get_config(args)

    def _get_config(self, args):
        if args.txt_model == 'bert':
            self._get_bert_config()

    def _get_bert_config(self):
        self.URL = 'beomi/kcbert'


class TrainConfig(CN):
    def __init__(self, args):
        super().__init__()
        self._get_base_config()
        self._get_optim_config(args)
        self._get_lr_scheduler_config(args)
        self._get_config_from_file(args)

    def _get_base_config(self):
        self.START_EPOCH    = 0
        self.EPOCHS         = 300
        self.START_EPOCH    = 0
        self.EPOCHS         = 300
        self.WARMUP_EPOCHS  = 20
        self.WEIGHT_DECAY   = 0.05
        self.BASE_LR        = 2e-5
        self.WARMUP_LR      = 2e-7
        self.MIN_LR         = 2e-6
        self.CLIP_GRAD      = 5.0

    def _get_optim_config(self, args):
        self.OPTIMIZER = OptimizerConfig(args)

    def _get_lr_scheduler_config(self, args):
        self.LR_SCHEDULER = LRSchedulerConfig(args)
        
    def _get_config_from_file(self, args):
        self.merge_from_file(args.train_cfg)

class OptimizerConfig(CN):
    def __init__(self, args):
        super().__init__()
        opt_lower = args.optim.lower()
        if opt_lower == 'sgd':
            self._sgd_config()
        elif opt_lower == 'adamw':
            self._adamw_config()

    def _sgd_config(self):
        self.MOMENTUM = 0.9

    def _adamw_config(self):
        self.EPS = 1e-8
        self.BETAS = (0.9, 0.999)

class LRSchedulerConfig(CN):
    def __init__(self, args):
        super().__init__()
        lr_lower = args.lr_scheduler.lower()
        self._get_base_config()
        if lr_lower == 'cosine':
            self._get_cosine_config()
        elif lr_lower == 'linear':
            self._get_linear_config()
        elif lr_lower == 'step':
            self._get_step_config()

    def _get_base_config(self):
        self.DECAY_EPOCHS = 30

    def _get_cosine_config(self):
        self.WARMUP_PREFIX = True

    def _get_linear_config(self):
        pass
    
    def _get_step_config(self):
        self.DECAY_RATE = 0.1

