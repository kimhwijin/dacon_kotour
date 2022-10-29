from .kotuor import Kotour

def build_loader(config):
    if config.DATA.NAME == 'kotour':
        return Kotour.from_config(config) 

