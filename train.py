import argparse
from lib.model import Model
from lib.dataset import AFLW2000

import numpy as np
import os


def split(data_dir, val_split=0.2):
    images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))

    # images = [os.path.join(data_dir, x) for x in all_files if x.endswith('.jpg')]
    np.random.seed(322)
    np.random.shuffle(images)
    split_idx = int(len(images)*val_split)
    return images[split_idx:], images[:split_idx]


def main(opts):
    model = Model(66, opts.size)
    model.model.summary()

    train_list, val_list = split(opts.data)
    print()
    print()
    print(len(train_list))

    train_dataset = AFLW2000(train_list, batch_size=opts.bs, input_size=opts.size)
    val_dataset = AFLW2000(val_list, batch_size=opts.bs, input_size=opts.size)

    chkpt_name = f'model_size{opts.size}_e{opts.epoch}_lr{opts.lr:.01E}.h5'
    model.train(chkpt_name, train_dataset, val_dataset, opts.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='DATA',
                        type=str, required=True)
    parser.add_argument('--lr', help='LR',
                        default=1e-3, type=float)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    parser.add_argument('--epoch', help='Train duration',
                        default=30, type=int)
    parser.add_argument('--bs', help='BS',
                        default=64, type=int)
    parser.add_argument('--output', help='Save every N epoch',
                        default='.', type=str)

    args = parser.parse_args()
    main(args)