import argparse
from lib.model import Model
from lib.dataset import AFLW2000

import numpy as np
import os


def split(data_dir, val_split=0.2):
    all_files = os.listdir(data_dir)
    images = [os.path.join(data_dir, x) for x in all_files if x.endswith('.jpg')]
    np.random.seed(322)
    np.random.shuffle(images)
    split_idx = int(len(images)*val_split)
    return images[split_idx:], images[:split_idx]


def main(opts):
    model = Model(66, 64)
    model.model.summary()

    train_list, val_list = split(opts.data)

    train_dataset = AFLW2000(train_list)
    val_dataset = AFLW2000(val_list)

    model.train('m.h5', train_dataset, val_dataset)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='input data',
                        required=True, type=str)
    # parser.add_argument('--output', help='save to fld',
    #                     required=True, type=str)
    # parser.add_argument('--weights', help='DATA',
    #                     default='weights/checkpoint.pth', type=str)
    # parser.add_argument('--size', help='Input image size',
    #                     default=224, type=int)
    # parser.add_argument('--force_cpu', help='Use only cpu',
    #                     action="store_true")


    args = parser.parse_args()
    main(args)