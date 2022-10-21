from torch import nn
import torch
from transformers import BertModel, BertTokenizer, ViTModel

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        img_model = ViTModel.from_pretrained(config.MODEL.IMAGE.URL)
        txt_model = BertModel.from_pretrained(config.MODEL.TEXT.URL)

        self.img_embedding = img_model.embeddings
        self.txt_embedding = txt_model.embeddings
        self.get_attn_mask = txt_model.get_extended_attention_mask

        self.encoder = txt_model.encoder
        self.pooler = txt_model.pooler

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
        txt_embed = self.txt_embedding(token_ids)

        x = torch.cat((img_embed, txt_embed), axis=1)
        attn_masks = self.get_attn_mask(attn_masks, attn_masks.shape, attn_masks.device)
        x = self.encoder(x, attn_masks)[0]
        x = self.pooler(x)
        x = self.head(x)
        return x


