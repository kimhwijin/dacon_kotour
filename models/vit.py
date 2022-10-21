from transformers import ViTModel

def get_vit(config):
    vit = ViTModel.from_pretrained(config.MODEL.IMAGE.URL)
    return vit