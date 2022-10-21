from timm.data import auto_augment
import albumentations as A
from PIL import Image, ImageOps
import numpy as np


auto_aug_policy = {
    'invert': [('Invert', 1.0, 9)],
    'solarize': [('Solarize', 1.0, 7), ('Equalize', 1.0, 5)],
    'gray': [('Color', 1.0, 0)]
}
aug_op = {
    'invert': [auto_augment.AugmentOp(*policy) for policy in auto_aug_policy['invert']],
    'solarize': [auto_augment.AugmentOp(*policy) for policy in auto_aug_policy['solarize']],
    'gray': [auto_augment.AugmentOp(*policy) for policy in auto_aug_policy['gray']],
    
    'elastic_transform': [A.ElasticTransform(p=1.0), A.ShiftScaleRotate(p=1.0)]
}

def horizontal_flip(img:Image):
    return ImageOps.mirror(img)

def gray_scale(img:Image):
    for op in aug_op['gray']:
        img = op(img)
    return img

def invert(img:Image):
    for op in aug_op['invert']:
        img = op(img)
    return img

def solarize(img:Image):
    for op in aug_op['solarize']:
        img = op(img)
    return img

def elastic_transform(img:Image):
    img = np.array(img)
    for op in aug_op['elastic_transform']:
        img = op(image=img)['image']
    return Image.fromarray(img)

def identity(img:Image):
    return img