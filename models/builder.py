from .model import Model
from torch import nn

def build_model(config):
    return Model(config)