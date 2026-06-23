import yaml
import argparse
from data.test import test
from train_spbf import train
import os


def parse_arguments():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-f', required=False, help='Path to a config.yaml file', type=os.path.abspath)
    parser.add_argument('--inference', action='store_true')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.test:
        test()
    elif args.train:
        if not args.f:
            raise RuntimeError('No path to config file provided')
        
        with open(args.f, 'r') as f:
            config = yaml.full_load(f)
        train(config)
    elif args.inference:
        pass
    else:
        raise RuntimeError('No known argument provided')




if __name__ == "__main__":
    main()
