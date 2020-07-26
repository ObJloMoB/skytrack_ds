import argparse
import cv2
import face_recognition
import numpy as np
from lib.model import Model
from math import cos, sin


def extend_crop(bbox, scale=2.0):
    cy, cx = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    new_h, new_w = (bbox[2] - bbox[0])*scale, (bbox[1] - bbox[3])*scale
    print(new_h, new_w)
    new_h = min([new_h, new_w])
    new_w = new_h
    print(new_h, new_w)
    new_bbox = np.array([cy - new_h/2, cx - new_w/2, cy + new_h/2, cx + new_w/2], dtype=np.int32)
    new_bbox = np.clip(new_bbox, 0, np.max(new_bbox))
    print(new_bbox)

    return new_bbox


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
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
    # replace with imutils
    cap = cv2.VideoCapture(opts.input)

    model = Model(66, opts.size)
    model.load(opts.weights)

    while True:
        _, img = cap.read()

        inp_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(inp_img)

        for (top, right, bottom, left) in face_locations:
            # crop = img[top:bottom, left:right]
            # cv2.imshow('crop', crop)

            top, left, bottom, right = extend_crop([top, right, bottom, left])
            crop = img[top:bottom, left:right]
            # cv2.imshow('new_crop', crop)

            crop = cv2.resize(crop, (224, 224))
            # cv2.imshow('resize', crop)
            input_img = np.asarray(crop, dtype=np.float32) / 255.0
            normed_img = (input_img - [0.5, 0.5, 0.5]) / [0.25, 0.25, 0.25]
            normed_img = np.expand_dims(normed_img, 0)
            res = model.test_online(normed_img)
            print(res)

            img = draw_axis(img, *res, tdx=left, tdy=top, size=100)


            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        # cv2.imshow('img', img)
        # if cv2.waitKey(1) == 27:
        #     break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input resiurce',
                        required=True, type=str)
    parser.add_argument('--weights', help='chkpt',
                        default='model_size224_e30_lr1.0E-05.h5', type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)


    args = parser.parse_args()
    main(args)