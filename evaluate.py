import argparse
import cv2
import face_recognition
import numpy as np
from lib.model import Model
from math import cos, sin
from time import time

from lib.dataset import AFLW2000
from train import split
from demo import draw_axis


def main(opts):
    model = Model(66, opts.size)
    model.model.summary()
    model.load(opts.weights)

    train_list, val_list = split(opts.data)
    val_dataset = AFLW2000(val_list, batch_size=1, input_size=opts.size)

    err, times = [], []
    for idx, (x, y) in enumerate(val_dataset.data_generator()):
        print(f'{idx}/{val_dataset.epoch_steps}')

        t1 = time()
        res = model.test_online(x)
        times.append(time()-t1)
        ypr = np.array(y)[:, 0, 1]
        err.append(abs(ypr-res))

        print(f'YPR: {np.mean(np.array(err), axis=0)}')
        print(f'TIME: {np.mean(times)}')
        if idx == val_dataset.epoch_steps:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='input resiurce',
                        required=True, type=str)
    parser.add_argument('--weights', help='chkpt',
                        default='model_size224_e30_lr1.0E-05.h5', type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    args = parser.parse_args()
    main(args)
