from config import Config
import argparse
import os
from models.builder import build_model

def parse_option():
    parser = argparse.ArgumentParser("Parsing Method")

    parser.add_argument('--output_path', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--num_class', type=int)

    parser.add_argument('--img_model', type=str, default='vit')
    parser.add_argument('--txt_model', type=str, default='bert')
    parser.add_argument('--model_cfg', type=str, default=os.path.join('.', 'configs', 'model', 'kcbert_vit_base_patch16_224.yaml'))
    parser.add_argument('--train_cfg', type=str, default=os.path.join('.', 'configs', 'training.yaml'))
    
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')

    args, _ = parser.parse_known_args()
    config = Config.from_args(args)
    return args, config


if __name__ == '__main__':
    args, config = parse_option()
    # model = build_model(config)
