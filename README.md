# Тестовое

## Задание 1


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
