import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        return _cosine(config, optimizer, n_iter_per_epoch)



def _cosine(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
        # t_mul=1.,
        lr_min=config.TRAIN.MIN_LR,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
        warmup_prefix=config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX,
    )
    return lr_scheduler