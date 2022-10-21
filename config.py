import os
from yacs.config import CfgNode as CN

def _check_args(name, args):
    if hasattr(args, name) and eval(f'args.{name}'):
        return True
    else:
        return False

class Config():
    
    @classmethod
    def from_args(cls, args):
        config          = cls._get_base_config(args)
        config.DATA     = DataConfig.from_args(args)
        config.MODEL    = ModelConfig.from_args(args)
        config.TRAIN    = TrainConfig.from_args(args)
        config.freeze()
        return config

    @staticmethod
    def _get_base_config(args):
        config              = CN()
        config.SEED         = 0                                     if not _check_args('seed', args) else args.seed
        config.TAG          = 'default'                             if not _check_args('tag', args) else args.tag
        config.OUTPUT       = os.path.join('./results', config.TAG) if not _check_args('output', args) else args.output
        config.NUM_WORKERS  = 0                                     if not _check_args('num_workers', args) else args.num_workers
        return config
        


class DataConfig():
    @staticmethod
    def from_args(args):
        DATA = CN()
        DATA.BATCH_SIZE  = 128                  if not _check_args('batch_size', args) else args.batch_size
        DATA.PATH        = './data'             if not _check_args('data_path', args) else args.data_path
        DATA.NAME        = 'kotour'             if not _check_args('data_name', args) else args.data_name
        DATA.NUM_CLASS   = 128                  if not _check_args('num_class', args) else args.num_class
        DATA.SPLIT       = 'stratified_kfold'   if not _check_args('data_split', args) else args.data_split
        DATA.TRAIN_PATH  = './preprocessed'     if not _check_args('train_data_path', args) else args.train_data_path
        return DATA

class ModelConfig():
    @classmethod
    def from_args(cls, args):
        MODEL               = cls._get_base_config()
        MODEL.IMAGE         = ImageModelConfig.from_args(args)
        MODEL.TEXT          = TextModelConfig.from_args(args)
        if _check_args('model_cfg', args):
            MODEL.merge_from_file(args.model_cfg)
        return MODEL

    @staticmethod
    def _get_base_config():
        MODEL               = CN()
        MODEL.SCALE         = 'base'
        MODEL.HIDDEN_DIM    = 768
        MODEL.MAX_SEQ       = 300
        return MODEL

class ImageModelConfig():
    @classmethod
    def from_args(cls, args):
        if args.img_model == 'vit':
            IMAGE = cls._get_vit_config()
        else:
            IMAGE = cls._get_vit_config()
        return IMAGE

    @staticmethod
    def _get_vit_config():
        IMAGE = CN()
        IMAGE.NAME = 'vit'
        IMAGE.SIZE = 224
        IMAGE.PATCH = 16
        IMAGE.MAX_SEQ = 197
        IMAGE.URL = 'google/vit-base-patch16-224'
        return IMAGE

class TextModelConfig():
    @classmethod
    def from_args(cls, args):
        if args.txt_model == 'bert':
            TEXT = cls._get_bert_config()
        else:
            TEXT = cls._get_bert_config()
        return TEXT

    @staticmethod
    def _get_bert_config():
        TEXT = CN()
        TEXT.NAME = 'kcbert'
        TEXT.URL = 'beomi/kcbert'
        return TEXT

class TrainConfig():
    @classmethod
    def from_args(cls, args):
        TRAIN               = cls._get_base_config()
        TRAIN.OPTIMIZER      = OptimizerConfig.from_args(args)
        TRAIN.LR_SCHEDULER  = LRSchedulerConfig.from_args(args)
        if _check_args('train_cfg', args):
            TRAIN.merge_from_file(args.train_cfg)
        return TRAIN

    @staticmethod
    def _get_base_config():
        TRAIN = CN()
        TRAIN.START_EPOCH    = 0
        TRAIN.EPOCHS         = 300
        TRAIN.START_EPOCH    = 0
        TRAIN.EPOCHS         = 300
        TRAIN.WARMUP_EPOCHS  = 20
        TRAIN.WEIGHT_DECAY   = 0.05
        TRAIN.BASE_LR        = 2e-5
        TRAIN.WARMUP_LR      = 2e-7
        TRAIN.MIN_LR         = 2e-6
        TRAIN.CLIP_GRAD      = 5.0
        return TRAIN

class OptimizerConfig():

    @classmethod
    def from_args(cls, args):
        OPTIMIZER = CN()
        opt_lower = args.optim.lower()
        if opt_lower == 'sgd':
            OPTIMIZER = cls._sgd_config(OPTIMIZER)
        elif opt_lower == 'adamw':
            OPTIMIZER = cls._adamw_config(OPTIMIZER)
        return OPTIMIZER

    @staticmethod
    def _sgd_config(OPTIMIZER):
        OPTIMIZER.NAME = 'sgd'
        OPTIMIZER.MOMENTUM = 0.9
        return OPTIMIZER

    @staticmethod
    def _adamw_config(OPTIMIZER):
        OPTIMIZER.NAME = 'adamw'
        OPTIMIZER.EPS = 0.9
        OPTIMIZER.BETAS = (0.9, 0.999)
        return OPTIMIZER

class LRSchedulerConfig():

    @classmethod
    def from_args(cls, args):
        lr_lower = args.lr_scheduler.lower()
        LR_SCHEDULER = cls._get_base_config()
        if lr_lower == 'cosine':
            LR_SCHEDULER = cls._get_cosine_config(LR_SCHEDULER)
        elif lr_lower == 'linear':
            LR_SCHEDULER = cls._get_linear_config(LR_SCHEDULER)
        return LR_SCHEDULER

    @staticmethod
    def _get_base_config():
        LR_SCHEDULER = CN()
        LR_SCHEDULER.DECAY_EPOCHS = 30
        return LR_SCHEDULER

    @staticmethod
    def _get_cosine_config(LR_SCHEDULER):
        LR_SCHEDULER.NAME = 'cosine'
        LR_SCHEDULER.WARMUP_PREFIX = True
        return LR_SCHEDULER

    @staticmethod
    def _get_linear_config(LR_SCHEDULER):
        LR_SCHEDULER.NAME = 'linear'
        return LR_SCHEDULER
