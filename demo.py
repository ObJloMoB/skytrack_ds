import argparse
import cv2
import face_recognition
import numpy as np
from lib.model import Model
from math import cos, sin


def resize_center_crop(image, size):
    w, h = image.shape[:2]
    if w > h:
        res = cv2.resize(image, (224, int(w/h*224)))
    else:
        res = cv2.resize(image, (int(h/w*224), 224))
    w, h = res.shape[:2]
    cw, ch = int(w/2), int(h/2)
    crop = res[int(cw-size/2):int(cw+size/2), int(ch-size/2):int(ch+size/2)]

    return crop


def normalize(image):
    input_img = np.asarray(image, dtype=np.float32) / 255.0
    normed_img = (input_img - np.array([0.5, 0.5, 0.5])) / np.array([0.25, 0.25, 0.25])
    return normed_img
    

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = -1*roll * np.pi / 180

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
    cap = cv2.VideoCapture(opts.input)

    model = Model(66, opts.size)
    model.load(opts.weights)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('data/res.mp4', fourcc, 10, (640, 480))

    while True:
        _, img = cap.read()

        # Предикт лица (по дефолту тут HOG)
        inp_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(inp_img)

        for (top, right, bottom, left) in face_locations:
            # Расширяем ббокс и смещаем вертикально
            bbox_width = abs(bottom - top)
            bbox_height = abs(right - left)
            left -= int(2 * bbox_width / 4)
            right += int(2 * bbox_width / 4)
            top -= int(3 * bbox_height / 4)
            bottom += int(bbox_height / 4)

            # Выход за пределы
            top = max(top, 0)
            left = max(left, 0)
            bottom = min(img.shape[0], bottom)
            right = min(img.shape[1], right)

            crop = img[top:bottom, left:right]

            # Ресайз по меньшей стороне и кроп от центра
            crop = resize_center_crop(crop, opts.size)

            # Нормализация
            normed_img = normalize(crop)
            imgs = []
            imgs.append(normed_img)

            # Предикт
            res = model.test_online(imgs)
            
            # Отрисовка
            img = draw_axis(img, *res, tdx=(left+right)/2, tdy=(top+bottom)/2, size=100)
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)

        out.write(img)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input resiurce',
                        required=True, type=str)
    parser.add_argument('--weights', help='chkpt',
                        default='data/headR_model_size224_e50_lr1.0E-05', type=str)
    parser.add_argument('--size', help='Input image size',
                        default=224, type=int)

    args = parser.parse_args()
    main(args)