import os
import numpy as np
import cv2
import scipy.io as sio
import random


class AFLW2000:
    def __init__(self, data_list, augment=False, batch_size=16, input_size=64):
        self.data_list = data_list
        self.batch_size = batch_size
        self.epoch_steps = len(data_list) // batch_size
        self.augment = augment
        self.input_size = input_size
        self.norm_params = {'mean': np.array([0.5, 0.5, 0.5]),
                            'std':  np.array([0.25, 0.25, 0.25])}
    
    # REWORK        
    def __get_ypr_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pre_pose_params = mat['Pose_Para'][0]
        pose_params = pre_pose_params[:3]
        return pose_params

    def __get_pt2d_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d
    # END REWORK 
    
    def __get_input_data(self, file_name):
        # Не перевожу в РГБ тк особого смысла нет, тк предобучение из бибилиотек не беру
        img = cv2.imread(file_name)
        pt2d = self.__get_pt2d_from_mat(file_name.replace('jpg', 'mat'))
        
        # Просто кроп
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])
        
        Lx = abs(x_max - x_min)
        Ly = abs(y_max - y_min)
        Lmax = max(Lx, Ly) * 1.5
        center_x = x_min + Lx // 2
        center_y = y_min + Ly // 2
        
        x_min = center_x - Lmax // 2
        x_max = center_x + Lmax // 2
        y_min = center_y - Lmax // 2
        y_max = center_y + Lmax // 2
        
        # Не выходить за границы картинок
        if x_min < 0:
            y_max -= abs(x_min)
            x_min = 0
        if y_min < 0:
            x_max -= abs(y_min)
            y_min = 0
        if x_max > img.shape[1]:
            y_min += abs(x_max - img.shape[1])
            x_max = img.shape[1]
        if y_max > img.shape[0]:
            x_min += abs(y_max - img.shape[0])
            y_max = img.shape[0]
        
        crop_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Загружаем углы
        pose = self.__get_ypr_from_mat(file_name.replace('jpg', 'mat'))
        pitch = pose[0] * 180.0 / np.pi
        yaw = pose[1] * 180.0 / np.pi
        roll = pose[2] * 180.0 / np.pi
        
        # Аугментация, включено только на трейне
        if self.augment:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                crop_img = cv2.flip(crop_img, 1)

            # # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                kernel = np.ones((5,5),np.float32)/25
                crop_img = cv2.filter2D(crop_img,-1,kernel)
        
        cont_labels = [yaw, pitch, roll]
        # Нужно для комбинации кросс энтропии и мсе
        bins = np.array(range(-99, 99, 3))
        bin_labels = np.digitize([yaw, pitch, roll], bins) - 1

        # Нормализация
        crop_img = np.asarray(cv2.resize(crop_img, (self.input_size, self.input_size))) / 255.0
        normed_img = (crop_img - self.norm_params['mean']) / self.norm_params['std']

        return normed_img, bin_labels, cont_labels
        
    def data_generator(self, shuffle=True):
        file_num = len(self.data_list)

        while True:
            if shuffle:
                np.random.shuffle(self.data_list)
            max_num = file_num - (file_num % self.batch_size)
            for i in range(0, max_num, self.batch_size):
                batch_x = []
                batch_yaw = []
                batch_pitch = []
                batch_roll = []
                for j in range(self.batch_size):
                    img, bin_labels, cont_labels = self.__get_input_data(self.data_list[i + j])
                    batch_x.append(img)
                    batch_yaw.append([bin_labels[0], cont_labels[0]])
                    batch_pitch.append([bin_labels[1], cont_labels[1]])
                    batch_roll.append([bin_labels[2], cont_labels[2]])
                
                batch_x = np.array(batch_x, dtype=np.float32)
                batch_yaw = np.array(batch_yaw)
                batch_pitch = np.array(batch_pitch)
                batch_roll = np.array(batch_roll)

                yield (batch_x, [batch_yaw, batch_pitch, batch_roll])

    
    
