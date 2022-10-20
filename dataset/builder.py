from .kotuor import Kotour
def build_loader(config):
    train_df = Kotour.from_config(config)
    return train_df