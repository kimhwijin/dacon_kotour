from config import Config
import argparse
from lr_scheduler import build_scheduler
from models import build_model
from optimizers import build_optimizer
from dataset import build_loader
from train import predict_with_test, run_training
import numpy as np
import random, torch, sklearn
import os

def parse_option():
    parser = argparse.ArgumentParser("Parsing Method")

    # general
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workders', type=int)
    parser.add_argument('--black_out', type=bool)

    # data
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--data_split', type=str)

    # model
    parser.add_argument('--img_model', type=str, default='vit')
    parser.add_argument('--txt_model', type=str, default='koelectra')
    parser.add_argument('--model_cfg', type=str, default='./configs/model/koelectra_vit_base_patch16_224.yaml')
    parser.add_argument('--best_model', type=str, default='./results/best.pth')

    # train
    parser.add_argument('--train_cfg', type=str, default='./configs/training.yaml')
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')

    args, _ = parser.parse_known_args()
    config = Config.from_args(args)
    return args, config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    sklearn.random.seed(seed)

if __name__ == '__main__':
    print('args parse')
    args, config = parse_option()

    if not os.path.exists(config.OUTPUT):
        os.makedirs(config.OUTPUT)
    
    with open(f'{config.OUTPUT}/config.yaml', 'w') as f:
        f.write(config.dump())

    set_seed(config.SEED)
    print('Build loader');train_dl, valid_dl, test_dl, label_encoder = build_loader(config)    
    print('Build model');model = build_model(config)
    print('Build optimizer');optimizer = build_optimizer(config, model)
    print('Build scheduler');scheduler = build_scheduler(config, optimizer, len(train_dl))
    print('Running');run_training(config, model, train_dl, valid_dl, optimizer, scheduler)
    print('Predicting');predict_with_test(config, model, test_dl, label_encoder)
    print('Finish!')
    print(f'Check {config.DATA.PATH}/sample_submission.csv')
    