from transformers import AutoTokenizer, ElectraTokenizer
from kobert_transformers.tokenization_kobert import KoBertTokenizer
def get_tokenizer(config):
    if config.MODEL.TEXT.NAME == 'kcbert':
        return AutoTokenizer.from_pretrained(config.MODEL.TEXT.URL)
    elif config.MODEL.TEXT.NAME == 'kobert':
        return KoBertTokenizer.from_pretrained(config.MODEL.TEXT.URL)
    elif 'electra' in config.MODEL.TEXT.NAME:
        return ElectraTokenizer.from_pretrained(config.MODEL.TEXT.URL)