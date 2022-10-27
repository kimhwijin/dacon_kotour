from transformers import BertModel, ElectraModel
from torch import nn

class TextModel():
    @staticmethod
    def get_embedding_and_encoder(config):
        if 'bert' in config.MODEL.TEXT.NAME:
            txt_model = BertModel.from_pretrained(config.MODEL.TEXT.URL)
            return txt_model.embeddings, txt_model.encoder, txt_model.get_extended_attention_mask
        elif 'electra' in config.MODEL.TEXT.NAME:
            txt_model = ElectraModel.from_pretrained(config.MODEL.TEXT.URL)
            return txt_model.embeddings, txt_model.encoder, txt_model.get_extended_attention_mask
    