from transformers import AutoTokenizer

def get_tokenizer(config):
    if config.MODEL.TEXT.NAME == 'kcbert':
        return AutoTokenizer.from_pretrained(config.MODEL.TEXT.URL)
    