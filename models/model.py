from torch import nn
import torch
from transformers import BertModel, ViTModel
from .text import TextModel
from .img import ImgModel
class Model(nn.Module):
    def __init__(self, config, output_attentions=False):
        super().__init__()

        self.out_attns = output_attentions
        
        self.img_embedding = ImgModel.get_embeddings(config)
        self.txt_embedding, self.encoder, self.get_attn_mask = TextModel.get_embedding_and_encoder(config)

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
        output = self.encoder(x, attn_masks, output_attentions=self.out_attns)
        x, attn_maps = output.last_hidden_state, output.attentions if self.out_attns else None
        x = x[:, 0]
        x = self.head(x)
        return x, attn_maps


