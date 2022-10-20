from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from .utils import _stratified_kfold
from .image_aug import horizontal_flip, gray_scale, invert, solarize, elastic_transform, identity
from PIL import Image
from .tokenizer import get_tokenizer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
import torch




class KotourDataset(Dataset):
    def __init__(self, config, df, train=True):
        self.config = config
        self.df = df
        self.train = train
        self.transform = A.Compose([
            A.Resize(config.MODEL.IMAGE.SIZE, config.MODEL.IMAGE.SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2(),
        ])
    def __getitem__(self, i):
        row = self.df.iloc[i]

        #Image
        image_path = row['img_path']
        image = io.imread(image_path)
        transformed_image = self.image_transform(image=image)['image']

        #Text
        input_ids = torch.Tensor(row['input_ids'])
        attn_mask = torch.Tensor([1 for _ in range(self.config.MODEL.IMAGE.MAX_SEQ)]+row['attn_mask'], dtype=torch.long)[:300]

        #Label
        if self.train:
            label = row['label']
        else:
            label = None

        return transformed_image, input_ids, attn_mask, label


class Kotour():
    @classmethod
    def from_config(cls, config):
        train_df, test_df           = cls._get_dataframe(config)
        train_df, test_df           = cls._preprocess_dataframe(config, train_df, test_df)
        train_df                    = cls._data_augmentation(config, train_df)
        train_df, valid_df          = cls._split_dataframe(config, train_df)
        train_df, valid_df, test_df = cls._convert_text_to_ids(config, train_df, valid_df, test_df)
        train_ds = KotourDataset(config, train_df)
        valid_ds = KotourDataset(config, valid_df)
        test_ds = KotourDataset(config, test_df, False)
        return train_ds, valid_ds, test_ds

    @staticmethod
    def _get_dataframe(config):
        train_df = pd.read_csv(config.DATA.PATH + '/train.csv')
        test_df = pd.read_csv(config.DATA.PATH +'/test.csv')
        return train_df, test_df

    @staticmethod
    def _preprocess_dataframe(config, train_df, test_df):
        train_df['img_path'] = train_df['img_path'].map(lambda t: config.DATA.PATH + '/image/train/' + t.split('/')[-1])
        test_df['img_path'] = test_df['img_path'].map(lambda t: config.DATA.PATH + '/image/test/' + t.split('/')[-1])
        le = LabelEncoder()
        le.fit(train_df['cat3'].values)
        train_df['label'] = le.transform(train_df['cat3'].values)
        return train_df, test_df

    @staticmethod
    def _split_dataframe(config, train_df):
        train_index, valid_index = _stratified_kfold(config, range(len(train_df['id'])), train_df['label'].values)
        return train_df.iloc[train_index], train_df.iloc[valid_index]
    
    @classmethod
    def _data_augmentation(cls, config, train_df):
        if not os.path.exists(config.DATA.TRAIN_PATH + '/train.csv'):
            os.mkdir(config.DATA.TRAIN_PATH);os.mkdir(config.DATA.TRAIN_PATH+'/train')
            train_df = cls._image_augmentation(config, train_df)
        train_df = pd.read_csv(config.DATA.TRAIN_PATH + '/train.csv')
        return train_df

    @staticmethod
    def _convert_text_to_ids(config, train_df, valid_df, test_df):
        tokenizer = get_tokenizer(config)
        def _tokenize(df, label=True):
            _toked = df['overview'].map(
                lambda txt: 
                tokenizer(txt, max_length=config.DATA.MAX_SEQ - config.MODEL.IMAGE.MAX_SEQ, padding='max_length')
            )
            df['input_ids'] = _toked.map(lambda d: d['input_ids'])
            df['attn_masks'] = _toked.map(lambda d: d['attention_mask'])
            if label:
                df = df[['id', 'img_path', 'input_ids', 'attn_masks', 'label']]
            else:
                df = df[['id', 'img_path', 'input_ids', 'attn_masks']]
            return df
        train_df = _tokenize(train_df)
        valid_df = _tokenize(valid_df)
        test_df = _tokenize(test_df, False)
        return train_df, valid_df, test_df



    @staticmethod
    def _image_augmentation(config, train_df):
        path = config.DATA.TRAIN_PATH

        from collections import Counter, OrderedDict
        c = Counter(train_df['label'])
        train_df['label_count'] = train_df['label'].map(lambda l: c[l])
        
        color_augs = [
            ('g', gray_scale),
            ('i', invert),
            ('s', solarize)
        ]
        transform_augs = [
            ('_', identity),
            ('t', elastic_transform)
        ]
        flip_augs = [
            ('_', identity),
            ('f', horizontal_flip)
        ]
        
        _12times_label = 2
        _8times_label = 20
        _4times_label = 60
        _2times_label = 130

        def _ntimes(train_df, bool_index, t_augs, f_augs, c_augs):
            aug_df = train_df[bool_index].copy()
            for t_name, t_aug in t_augs:
                for f_name, f_aug in f_augs:
                    for c_name, c_aug in c_augs:
                        aug_name = "{}{}{}".format(t_name, f_name, c_name)
                        _temp_aug_df = aug_df.copy()

                        for img_id, img_path in zip(_temp_aug_df['id'].values, _temp_aug_df['img_path'].values):
                            img = Image.open(img_path)
                            img = t_aug(f_aug(c_aug(img)))
                            img.save("{}/train/{}.jpg".format(path, img_id+aug_name))
                        
                        _temp_aug_df['id'] = _temp_aug_df['id'].map(lambda i: i+aug_name)
                        _temp_aug_df['img_path'] = _temp_aug_df['id'].map(lambda i: "{}/train/{}.jpg".format(path, i))
                        train_df = pd.concat([train_df, _temp_aug_df])

            return train_df

        # 12 times
        train_df = _ntimes(
            train_df, 
            train_df['label_count'] == _12times_label, 
            transform_augs, 
            flip_augs, 
            color_augs
        )
        # 8 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _8times_label) * (train_df['label_count'] > _12times_label), 
            transform_augs, 
            flip_augs, 
            color_augs[:2]
        )
        # 4 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _4times_label) * (train_df['label_count'] > _8times_label), 
            transform_augs[:1],
            flip_augs, 
            color_augs[:2]
        )
        # 2 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _2times_label) * (train_df['label_count'] > _4times_label), 
            transform_augs[:1],
            flip_augs[1:],
            color_augs[:2]
        )
        train_df.to_csv("{}/train.csv".format(path, 'train.csv'), index=False)
        return train_df