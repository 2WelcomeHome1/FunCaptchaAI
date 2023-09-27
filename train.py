import os, warnings
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import cv2 as cv
import numpy as np

warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_addons as tfa
import mlflow
from model import Model
from test import test_model

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.compat.v1.config.experimental.list_physical_devices("GPU") else "OFF" }')


class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None): 
        val_accuracy = logs["val_accuracy"]
        if (val_accuracy >= self.point):
            self.model.stop_training = True


class Train ():
    def __init__(self, lr_rate, epoch, batch_size, input_size, num_classes=6):
        self.lr_rate = lr_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_classes = num_classes
        self.le = LabelEncoder()

    def load_image(self, imagepath, input_size):
        image = cv.imread(imagepath, 1)
        if input_size != (110, 110): image = cv.resize(image, input_size) #изменяем размер
        return cv.cvtColor(image,cv.COLOR_BGR2GRAY) ## переводим в оттенки серого

    def load_data(self, path, input_size):
        X_list, y_list = [], []
        for file in os.listdir(path):
            imagepath = str(path+file)
            X_list.append(self.load_image(imagepath, input_size)) if file.split('.')[1] =='png' else y_list.append(open(imagepath, 'r').read())
        
        return np.array(X_list), np.array(y_list)
    
    def create_train_test_data(self):
        X_train, y_train = self.load_data(path = "./train/", input_size = self.input_size)
        x_valid, y_valid = self.load_data(path = "./valid/", input_size = self.input_size)
        x_test, y_test = self.load_data(path = "./test/", input_size = self.input_size)

        mlflow.log_params({'input_size': self.input_size})

        y_train = to_categorical(self.le.fit_transform(y_train), self.num_classes)
        y_valid = to_categorical(self.le.fit_transform(y_valid), self.num_classes)
        # y_test = to_categorical(self.le.fit_transform(y_test), self.num_classes)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_valid.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        return X_train, x_valid, x_test, y_train, y_valid, y_test

    def test_model(self, model, x_test, y_test):
        model.load_weights('saved_models/best_model.h5')

        test_accuracy = test_model(model, x_test, y_test)
        mlflow.log_metrics({'test_accuracy': test_accuracy})


    def train(self, summary: bool = True):
        mlflow.tensorflow.autolog()

        X_train, x_valid, x_test, y_train, y_valid, y_test = self.create_train_test_data()
        model = Model(X_train, self.num_classes, self.lr_rate).create_model()
        if summary: model.summary()

        Checkpoint = tf.keras.callbacks.ModelCheckpoint("saved_models/checkpoints/model-{epoch:02d}-{val_accuracy:.4f}.h5", save_best_only=True, monitor="val_accuracy", save_weights_only=True)
        Best_Checkpoint = tf.keras.callbacks.ModelCheckpoint("saved_models/best_model.h5", save_best_only=True, save_weights_only=True, monitor="val_accuracy")

        model.fit(X_train, y_train, batch_size=self.batch_size,
                            epochs=self.epoch, validation_data=(x_valid, y_valid), callbacks = [Checkpoint, Best_Checkpoint])

        self.test_model(model, x_test, y_test)

