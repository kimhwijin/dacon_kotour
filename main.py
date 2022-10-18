from ast import parse
from config import Config
import argparse

def parse_option():
    parser = argparse.ArgumentParser("Parsing Method")

    parser.add_argument('--output_path', type=str)
    parser.add_argument('--tag', type=str)
    parser.add_argument('--seed', type=int)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--num_class', type=int)

    parser.add_argument('--model_cfg', type=str)
    parser.add_argument('--train_cfg', type=str)

    args, _ = parser.parse_known_args()
    config = Config.get_instance(args)
    return args, config


if __name__ == '__main__':
    args, config = parse_option()
    print(args, config)