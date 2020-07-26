import argparse
import cv2
import face_recognition
import numpy as np
from lib.model import Model
from math import cos, sin
from lib.dataset import AFLW2000
from train import split

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = (yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis pointing down drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img

def main(opts):
    train_list, val_list = split(opts.data)
    train_dataset = AFLW2000(val_list, batch_size=1, input_size=opts.size)

    max_show = 10
    for idx, (x, y) in enumerate(train_dataset.data_generator()):
        img = x[0]*[0.25, 0.25, 0.25] + [0.5, 0.5, 0.5]
        img = (img*255).astype(np.uint8)
        yaw, pitch, roll = y[0][0][1], y[1][0][1], y[2][0][1]
        print(idx, yaw, pitch, roll)
        img = draw_axis(img, pitch, yaw, roll, tdx=opts.size/2, tdy=opts.size/2, size=100)
        cv2.imwrite(f'data/{idx}.jpg', img)
        if idx == max_show:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='input resiurce',
                        required=True, type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)


    args = parser.parse_args()
    main(args)