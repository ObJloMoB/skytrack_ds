import argparse
from lib.model import Model
from lib.dataset import AFLW2000

import numpy as np
import os


def split(data_dir, val_split=0.2):
    # Не лучшее решение дял сплита, тк отличается на разных машинах
    images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                images.append(os.path.join(root, file))

    # Но каждый ран фиксированный. Сплит стандартный, без тестовой, только с валом
    np.random.seed(322)
    np.random.shuffle(images)
    split_idx = int(len(images)*val_split)
    return images[split_idx:], images[:split_idx]


def main(opts):
    model = Model(66, opts.size)

    # Если есть предобучение, то возьмем
    if opts.pretrain is not None:
        print(f'Initial weights from {opts.pretrain}')
        model.load(opts.pretrain)

    train_list, val_list = split(opts.data)

    # Обучающий лоадер с аугментациями, но там их не сильно много
    train_dataset = AFLW2000(train_list, augment=True, batch_size=opts.bs, input_size=opts.size)
    val_dataset = AFLW2000(val_list, batch_size=opts.bs, input_size=opts.size)

    # Учим
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
    parser.add_argument('--pretrain', help='load weights',
                        default=None)

    args = parser.parse_args()
    main(args)