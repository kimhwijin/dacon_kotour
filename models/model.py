from multiprocessing import pool
from numpy import polyder
from torch import nn
import torch
from transformers import BertModel, ViTModel, ElectraModel
from .text import TextModel
from .img import ImgModel
class Model(nn.Module):
    def __init__(self, config, output_attentions=False):
        super().__init__()

        self.out_attns = output_attentions
        
        img_model = ViTModel.from_pretrained(config.MODEL.IMAGE.URL)
        self.img_embedding = img_model.embeddings
        if 'bert' in config.MODEL.TEXT.NAME:
            txt_model = BertModel.from_pretrained(config.MODEL.TEXT.URL)
        elif 'electra' in config.MODEL.TEXT.NAME:
            txt_model = ElectraModel.from_pretrained(config.MODEL.TEXT.URL)
        else:
            txt_model = BertModel.from_pretrained(config.MODEL.TEXT.URL)
            
        self.txt_embeddings, self.encoder, self.get_attn_mask = txt_model.embeddings, txt_model.encoder, txt_model.get_extended_attention_mask
        
        self.pooler = Pooler(config)

        self.transform = nn.Sequential(
            nn.Linear(config.MODEL.HIDDEN_DIM, config.MODEL.HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(config.MODEL.HIDDEN_DIM, eps=1e-12)
        )

        self.decoder = nn.Linear(config.MODEL.HIDDEN_DIM, config.DATA.NUM_CLASS, bias=False)
        bias = nn.Parameter(torch.zeros(config.DATA.NUM_CLASS))
        self.decoder.bias = bias
        self.head = nn.Sequential(
            self.transform,
            self.decoder
        )
        
    def forward(self, images, token_ids, attn_masks):
        img_embed = self.img_embedding(images)
        txt_embed = self.txt_embeddings(token_ids)

        x = torch.cat((img_embed, txt_embed), axis=1)
        attn_masks = self.get_attn_mask(attn_masks, attn_masks.shape, attn_masks.device)
        output = self.encoder(x, attn_masks, output_attentions=self.out_attns)
        x, attn_maps = output.last_hidden_state, output.attentions if self.out_attns else None
        x = self.pooler(x)
        x = self.head(x)
        return x, attn_maps


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.MODEL.HIDDEN_DIM, config.MODEL.HIDDEN_DIM)
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        first_token = hidden_state[:, 0]
        pooled_output = self.dense(first_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output
