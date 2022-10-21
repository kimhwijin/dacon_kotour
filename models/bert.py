from transformers import BertModel
from torch import nn

def get_bert(config):
    bert = BertModel.from_pretrained(config.MODEL.TEXT.URL)
    return bert