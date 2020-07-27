# Тестовое

## Задание 1
Все взято со статьи [Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925), у которой есть открытая [реализация](https://github.com/natanielruiz/deep-head-pose)

Тестировалось под:
- Ubuntu 18.04
- Cuda 10.0
- Python 3.6

[Ссылка](https://drive.google.com/file/d/13Uy1IlB_XSg6MYdT-83vZaed-pCfEPxo/view?usp=sharing) на веса. Их положить в папку `data`.

```sh
pip3 install -r requirements.txt
python3 demo.py --input /dev/video0
```
Результаты обучения на датасете 300W-LP (претрейн на ALWF2k)
![Гиф работы](https://github.com/ObJloMoB/skytrack_ds/blob/master/data/res.gif)

## Задание 2


## Задание 3
Итоговая вероятность просто отношение увидивших версию А к увидившим обе версии. События не зависимы.
P = 0.08 * 0.6/(0.08 * 0.6 + 0.04 * 0.4) = 0.75

## Задание 4
По заданию дано, что строчки в датафрейме и в матрице нампая совпадают, поэтому функция делается в две строчки, тк не нужны проверки на соответствия Id
```python
import pandas as pd
import numpy as np

def login_table(id_name_verified, id_password): 
  id_name_verified.drop(columns='Verified', inplace=True)
  id_name_verified['Password'] = id_password[:, 1]

id_name_verified = pd.DataFrame([[1, 'JohnDoe', True], [2, 'AnnFranklin', True]], columns=['Id', 'Login', 'Verified'])
id_password = np.array([[1, 123456789], [2, 987654321]], np.int32)
login_table(id_name_verified, id_password)
print(id_name_verified)
```
