import argparse
import cv2
import face_recognition
import numpy as np
from lib.model import Model
from math import cos, sin

from lib.dataset import AFLW2000
from train import split
from demo import draw_axis

def extend_crop(bbox, scale=2.0):
    cy, cx = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    new_h, new_w = (bbox[2] - bbox[0])*scale, (bbox[3] - bbox[1])*scale
    new_bbox = np.array([cy - new_h/2, cx - new_w/2, cy + new_h/2, cx + new_w/2], dtype=np.int32)


    return new_bbox


def main(opts):
    model = Model(66, opts.size)
    model.load(opts.weights)

    train_list, val_list = split(opts.data)
    val_dataset = AFLW2000(val_list, batch_size=1, input_size=opts.size)

    yaw_err, pitch_err, roll_err = [], [], []
    for idx, (x, y) in enumerate(val_dataset.data_generator()):
        print(idx)

        res = model.test_online(x)
        yaw, pitch, roll = y[0][0][1], y[1][0][1], y[2][0][1]
        print(yaw, pitch, roll)
        print(res)
        yaw_err.append(abs(yaw-res[0]))
        pitch_err.append(abs(pitch-res[1]))
        roll_err.append(abs(roll-res[2]))

        img = x[0]*[0.25, 0.25, 0.25] + [0.5, 0.5, 0.5]

        img = draw_axis(img, pitch, yaw, roll, tdx=opts.size*0.25, tdy=opts.size*0.25, size=100)
        # img = draw_axis(img, *res, tdx=opts.size*0.75, tdy=opts.size*0.75, size=100)

        cv2.imshow('img', img)
        cv2.waitKey(0)

        print(f'YAW: {np.mean(yaw_err)}')
        print(f'PITCH: {np.mean(pitch_err)}')
        print(f'ROLL: {np.mean(roll_err)}')



        if idx == val_dataset.epoch_steps:
            break

    print('##########################')
    print(f'YAW: {np.mean(yaw_err)}')
    print(f'PITCH: {np.mean(pitch_err)}')
    print(f'ROLL: {np.mean(roll_err)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='input resiurce',
                        required=True, type=str)
    parser.add_argument('--weights', help='chkpt',
                        default='model_size224_e50_lr1.0E-03.h5', type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)
    args = parser.parse_args()
    main(args)