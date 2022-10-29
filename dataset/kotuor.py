from numpy import bool_
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

from models.builder import build_model
from .utils import _stratified_kfold
from .image_aug import horizontal_flip, gray_scale, invert, solarize, elastic_transform, identity
from PIL import Image
from .tokenizer import get_tokenizer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from skimage import io
import torch
from tqdm import tqdm

class KotourDataset(Dataset):
    def __init__(self, config, df, train=True):
        self.config = config
        self.df = df
        self.train = train

        self.image_transform = A.Compose([
            A.Resize(config.MODEL.IMAGE.SIZE, config.MODEL.IMAGE.SIZE),
            A.RandomCrop(config.MODEL.IMAGE.SIZE, config.MODEL.IMAGE.SIZE, p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]) if train else A.Compose([
            A.Resize(config.MODEL.IMAGE.SIZE, config.MODEL.IMAGE.SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])

    def __getitem__(self, i):
        row = self.df.iloc[i]

        #Image ID
        img_id = row['id']

        #Image
        image_path = row['img_path']
        image = io.imread(image_path)
        image = self.image_transform(image=image)['image']

        #Text
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        attn_masks = [1] * self.config.MODEL.IMAGE.MAX_SEQ + row['attn_masks'][:self.config.MODEL.MAX_SEQ - self.config.MODEL.IMAGE.MAX_SEQ]
        
        if self.train and self.config.BLACK_OUT:
            torch.randint(0, )
            black_out_index = row['black_out']
            
            attn_masks[row['black_out']] = 0

        attn_masks = torch.tensor(attn_masks, dtype=torch.long)

        #Label
        if self.train:
            label = row['label']
            return img_id, image, input_ids, attn_masks, label
        else:
            return img_id, image, input_ids, attn_masks


    def __len__(self):
        return len(self.df)

class Kotour():
    @classmethod
    def from_config(cls, config):
        train_df, test_df           = cls._get_dataframe(config)
        train_df, test_df, le       = cls._preprocess_dataframe(config, train_df, test_df)
        train_df                    = cls._data_augmentation(config, train_df)
        train_df, test_df           = cls._convert_text_to_ids(config, train_df, test_df)
        train_df                    = cls._get_attention_score(config, train_df)
        train_df, valid_df          = cls._split_dataframe(config, train_df)
        train_ds, valid_ds, test_ds = KotourDataset(config, train_df), KotourDataset(config, valid_df), KotourDataset(config, test_df, False)
        train_dl, valid_dl, test_dl = cls._make_dataloader(config, train_ds, valid_ds, test_ds)
        return train_dl, valid_dl, test_dl, le

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
        return train_df, test_df, le

    @classmethod
    def _data_augmentation(cls, config, train_df):
        if not os.path.exists(config.DATA.TRAIN_PATH + '/augged_train.csv'):
            os.mkdir(config.DATA.TRAIN_PATH);os.mkdir(config.DATA.TRAIN_PATH+'/train')
            train_df = cls._image_augmentation(config, train_df)
        train_df = pd.read_csv(config.DATA.TRAIN_PATH + '/augged_train.csv')
        return train_df
    
    @staticmethod
    def _convert_text_to_ids(config, train_df, test_df):
        if not (os.path.exists(f"{config.DATA.TRAIN_PATH}/train.csv") and os.path.exists(f"{config.DATA.TRAIN_PATH}/test.csv")):
            print("    Convert overview to id sequences...")
            tokenizer = get_tokenizer(config)
            def _tokenize(df, label=True):
                _toked = df['overview'].map(
                    lambda txt: 
                    tokenizer(txt, max_length=config.MODEL.MAX_SEQ-config.MODEL.IMAGE.MAX_SEQ, padding='max_length')
                )
                df['input_ids'] = _toked.map(lambda d: d['input_ids'][:config.MODEL.MAX_SEQ-config.MODEL.IMAGE.MAX_SEQ])
                df['attn_masks'] = _toked.map(lambda d: d['attention_mask'][:config.MODEL.MAX_SEQ-config.MODEL.IMAGE.MAX_SEQ])
                if label:
                    df = df[['id', 'img_path', 'input_ids', 'attn_masks', 'label']]
                else:
                    df = df[['id', 'img_path', 'input_ids', 'attn_masks']]
                return df
            train_df = _tokenize(train_df)
            test_df = _tokenize(test_df, False)
            train_df.to_csv(f"{config.DATA.TRAIN_PATH}/train.csv", index=False)
            test_df.to_csv(f"{config.DATA.TRAIN_PATH}/test.csv", index=False)
            
        train_df = pd.read_csv(f"{config.DATA.TRAIN_PATH}/train.csv")
        test_df = pd.read_csv(f"{config.DATA.TRAIN_PATH}/test.csv")
        train_df['input_ids'] = train_df['input_ids'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        train_df['attn_masks'] = train_df['attn_masks'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        test_df['input_ids'] = test_df['input_ids'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        test_df['attn_masks'] = test_df['attn_masks'].apply(lambda x: list(map(int, x[1:-1].split(','))))
        return train_df, test_df

    @staticmethod
    def _get_attention_score(config, train_df):
        
        if not config.BLACK_OUT:
            return train_df
        
        if os.path.exists(f'{config.DATA.TRAIN_PATH}/black_out.csv'):
            train_df = pd.read_csv(f'{config.DATA.TRAIN_PATH}/black_out.csv')
            return train_df
        
        s = torch.load(config.BEST)
        device = torch.device(config.DEVICE)
        best_config = s['config']
        best_model = build_model(best_config).to(device)
        best_model.load_state_dict(s['model'])
        best_model = best_model.eval().to(device)


        train_ds = KotourDataset(config, train_df, train=False)
        train_dl = DataLoader(train_ds, shuffle=False, batch_size=config.DATA.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        pbar = tqdm(train_dl, total=len(train_dl), desc='    Calculating Attention Scores...')
        black_out = {
            "id": [],
            "black_out" : []
        }

        from .utils import _step, _get_score

        for data in pbar:
            img_ids, imgs, input_ids, attn_masks = data
            att_mats = torch.stack(_step(best_model, imgs.to(device), input_ids.to(device), attn_masks.to(device)))
            for j in range(imgs.size(0)):
                #att_mats : Layer Attn x Batch x Multi Head x Seq x Seq
                att_mat = att_mats[:,j,:,:,:]
                #att_mat : Layer Attn x Multi Head x Seq x Seq
                black_out["black_out"].append(_get_score(att_mat, device, config.MODEL.IMAGE.MAX_SEQ))
                black_out["id"].append(img_ids[j])

        black_out = pd.DataFrame(black_out)
        df = pd.merge(df, black_out, validate='one_to_one', on='id')
        df.to_csv(f"{config.DATA.TRAIN_PATH}/black_out.csv")
        return df


    @staticmethod
    def _split_dataframe(config, train_df):
        train_index, valid_index = _stratified_kfold(config, range(len(train_df['id'])), train_df['label'].values)
        return train_df.iloc[train_index], train_df.iloc[valid_index]

    @staticmethod
    def _make_dataloader(config, train_ds, valid_ds, test_ds):
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=config.DATA.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        valid_dl = DataLoader(valid_ds, shuffle=True, batch_size=config.DATA.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        test_dl  = DataLoader(test_ds, batch_size=config.DATA.BATCH_SIZE, num_workers=config.NUM_WORKERS)
        return train_dl, valid_dl, test_dl

    @staticmethod
    def _image_augmentation(config, train_df):
        path = config.DATA.TRAIN_PATH

        from collections import Counter
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

            for i in tqdm(range(len(aug_df)), "    Augmenting..."):
                row = aug_df.iloc[i]
                img = Image.open(row['img_path'])
                #
                for t_name, t_aug in t_augs:
                    for f_name, f_aug in f_augs:
                        for c_name, c_aug in c_augs:
                            aug_name = f"{t_name}{f_name}{c_name}"
                            augged_row = row.copy()

                            augged_row['id'] = augged_row['id']+aug_name
                            augged_row['img_path'] = f"{path}/train/{augged_row['id']}.jpg"

                            aug_img = t_aug(f_aug(c_aug(img)))
                            aug_img.save(f"{path}/train/{augged_row['id']}.jpg")
                            #append
                            aug_df.loc[len(aug_df)] = augged_row
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            return train_df

        print("    Augmenting 12 times")
        # 12 times
        train_df = _ntimes(
            train_df, 
            train_df['label_count'] == _12times_label, 
            transform_augs, 
            flip_augs, 
            color_augs
        )
        print("    Augmenting 8 times")
        # 8 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _8times_label) * (train_df['label_count'] > _12times_label), 
            transform_augs, 
            flip_augs, 
            color_augs[:2]
        )
        print("    Augmenting 4 times")
        # 4 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _4times_label) * (train_df['label_count'] > _8times_label), 
            transform_augs[:1],
            flip_augs, 
            color_augs[:2]
        )
        print("    Augmenting 2 times")
        # 2 times
        train_df = _ntimes(
            train_df, 
            (train_df['label_count'] <= _2times_label) * (train_df['label_count'] > _4times_label), 
            transform_augs[:1],
            flip_augs[1:],
            color_augs[:2]
        )
        train_df.to_csv("{}/augged_train.csv".format(path), index=False)
        return train_df