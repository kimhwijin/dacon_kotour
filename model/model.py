from torch import nn
import torch
from transformers import BertModel, BertTokenizer, ViTModel
_URL = {
    'vit' : {
        'base-patch16-224':'google/vit-base-patch16-224',
        'large-patch16-224':'google/vit-large-patch16-224',

        'base-patch16-384':'google/vit-base-patch16-384',
        'large-patch16-384':'google/vit-large-patch16-384',

        'base-patch32-224':'google/vit-base-patch32-224-in21k',
        'large-patch32-224':'google/vit-large-patch32-224-in21k',

        'base-patch32-384':'google/vit-base-patch32-384',
        'large-patch32-384':'google/vit-large-patch32-384',
    },
    'kcbert': {
        'base': 'beomi/kcbert-base',
        'large': 'beomi/kcbert-large'
    },
    'kobert':{
        'base': 'skt/kobert-base-v1'
    }
}

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        img_model_url = _URL[config.MODEL.IMAGE.NAME]["{}-patch{}-{}".format(config.MODEL.SCALE, config.MODEL.IMAGE.PATCH, config.MODEL.IMAGE.SIZE)]
        txt_model_url = _URL[config.MODEL.TEXT.NAME][config.MODEL.SCALE]
        
        img_model = ViTModel.from_pretrained(img_model_url)
        txt_model = BertModel.from_pretrained(txt_model_url)

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

        self.decoder = nn.Linear(config.MODEL.HIDDEN_DIM, config.NUM_CLASS, bias=False)
        bias = nn.Parameter(torch.zeros(config.NUM_CLASS))
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
        x = self.encoder(x, attn_masks)
        x = self.head(x)
        return x


