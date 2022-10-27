from transformers import AutoTokenizer, ElectraTokenizer

def get_tokenizer(config):
    if config.MODEL.TEXT.NAME == 'kcbert':
        return AutoTokenizer.from_pretrained(config.MODEL.TEXT.URL)
    if 'electra' in config.MODEL.TEXT.NAME:
        return ElectraTokenizer.from_pretrained(config.MODEL.TEXT.URL)