import argparse
import os

import yaml

from data.dataset import cacheDataSet
from data.test import test
from train import train

def parse_arguments():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('-f', required=False, type=os.path.abspath, help='Path to a config .yml file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.test:
        test()
    elif args.train:
        if not args.f:
            raise RuntimeError('No path to config file provided')
        with open(args.f, 'r') as handle:
            config = yaml.full_load(handle)
        print(f'Training {config["name"]}')
        train(config)
    elif args.inference:
        pass
    else:
        raise RuntimeError('No known argument provided')

if __name__ == '__main__':
    main()
    
