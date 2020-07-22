import argparse
import cv2
import face_recognition


def main(opts):
    # replace with imutils
    cap = cv2.VideoCapture(opts.input)

    while True:
        _, img = cap.read()

        inp_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(inp_img)


        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input resiurce',
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