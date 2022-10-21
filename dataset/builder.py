from .kotuor import Kotour

def build_loader(config):
    return Kotour.from_config(config) 

