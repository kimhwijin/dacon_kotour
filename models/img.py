from transformers import ViTModel

class ImgModel():
    @staticmethod
    def get_embeddings(config):
        if 'vit' in config.MODEL.IMAGE.NAME:
            img_model = ViTModel.from_pretrained(config.MODEL.IMAGE.URL)
            return img_model.embeddings