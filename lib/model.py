import tensorflow as tf
import tensorflow.keras as K
import os
import numpy as np
import cv2



def scheduler(epoch):
  if epoch < 25:
    return 1e-3
  else:
    return 1e-4



class Model:
    def __init__(self, class_num, input_size):
        self.class_num = class_num
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = np.array(self.idx_tensor, dtype=np.float32)
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred, alpha=1.0):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]

        sm_pred = K.backend.softmax(y_pred, 1)
        one_hot = K.backend.one_hot(K.backend.cast(bin_true, 'int32'), 66)
        cls_loss = K.losses.categorical_crossentropy(one_hot, sm_pred)
        # MSE loss
        pred_cont = K.backend.sum(sm_pred * self.idx_tensor, 1) * 3 - 99
        mse_loss = K.losses.MSE(cont_true, pred_cont)
        # Total loss
        # mse_loss = 0
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        
        feature = K.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
        feature = K.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = K.layers.Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
        feature = K.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = K.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = K.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = K.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
        feature = K.layers.MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        feature = K.layers.Flatten()(feature)
        feature = K.layers.Dropout(0.5)(feature)
        feature = K.layers.Dense(units=4096, activation=tf.nn.relu)(feature)
        
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        return model

    def train(self, 
              model_path, 
              train_dataset,
              val_dataset,
              max_epoches=20,
              lr=1e-4,):
        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        
        self.model.compile(optimizer=K.optimizers.Adam(lr=lr), loss=losses)

        self.model.summary()
        reducer = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        self.model.fit(x=train_dataset.data_generator(),
                       validation_data=val_dataset.data_generator(),
                       epochs=max_epoches,
                       callbacks=[reducer, ],
                       steps_per_epoch=train_dataset.epoch_steps,
                       validation_steps=val_dataset.epoch_steps,
                       verbose=1)

        # self.model.fit_generator(generator=train_dataset.data_generator(),
        #                             epochs=max_epoches,
        #                             steps_per_epoch=train_dataset.epoch_steps,
        #                             max_queue_size=10,
        #                             workers=1,
        #                             verbose=1)

        self.model.save(model_path)

    def load(self, path):
        self.model.load_weights(path)
            
    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x, batch_size=1, verbose=0)
        predictions = np.asarray(predictions)
        pred_cont_yaw = K.backend.sum(K.backend.softmax(predictions[0, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = K.backend.sum(K.backend.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = K.backend.sum(K.backend.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 99
        
        return K.backend.eval(pred_cont_yaw)[0], K.backend.eval(pred_cont_pitch)[0], K.backend.eval(pred_cont_roll)[0]



