import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Убираем из консоли служебную информацию от TensorFlow
import keras
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Dropout, Flatten
from keras.api.layers import Conv2D, MaxPooling2D
import cv2
import numpy as np
import tensorflow as tf
tf.config.run_functions_eagerly(True)


TRAIN_DIR_WOMEN = '/train/women'
TRAIN_DIR_MEN = '/train/men'
TEST_DIR_WOMEN = '/test/women'
TEST_DIR_MEN = '/test/men'
WORK_SIZE = (200, 200)
NUM_CLASSES = 2

def create_conv_model():
    input_shape = (WORK_SIZE[0],WORK_SIZE[1],1)
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(8, 8), activation='relu', input_shape=input_shape)) # Добавляем входной слой
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', input_shape=input_shape))  # Добавляем входной слой
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=input_shape))  # Добавляем входной слой
    model.add(MaxPooling2D(pool_size=((WORK_SIZE[0]-16+1)/5, (WORK_SIZE[0]-16+1)/5)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print(model.summary())
    return model

def load_dataset(ds_dir):

    women_train, men_train = load_ds_train(ds_dir)  # Загружаем тренировочные данные
    women_test, men_test = load_ds_test(ds_dir)  # Загружаем тестовые данные
    return women_train, men_train, women_test, men_test

def load_ds_train(ds_dir):
    women_train = []
    men_train = []

    tmp_dir = str(ds_dir + TRAIN_DIR_WOMEN)
    women_train, men_train = load_ds_image(women_train, men_train, tmp_dir, 1)

    tmp_dir = str(ds_dir + TRAIN_DIR_MEN)
    women_train, men_train = load_ds_image(women_train, men_train, tmp_dir, 0)

    return women_train, men_train


def load_ds_test(ds_dir):
    women_test = []
    men_test = []

    tmp_dir = str(ds_dir + TEST_DIR_WOMEN)
    women_test, men_test = load_ds_image(women_test, men_test, tmp_dir, 1)

    tmp_dir = str(ds_dir + TEST_DIR_MEN)
    women_test, men_test = load_ds_image(women_test, men_test, tmp_dir, 0)

    return women_test, men_test


def load_ds_image(x, y, dir, goodflag):
    tmp_dir = str(dir)
    filelist = os.listdir(tmp_dir)
    for i in filelist:
        tmp_img = load_img_from_file(str(tmp_dir + '/' + i), WORK_SIZE)
        x.append(tmp_img)
        y.append(int(goodflag))
    return x, y


def load_img_from_file(fname, imgsize=WORK_SIZE):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, imgsize)
    img = np.expand_dims(img, axis=2)
    return img

def learn_conv_model(model):
    ds_dir = "dataset"
    women_train, men_train, women_test, men_test = load_dataset(ds_dir)

    men_train = keras.utils.to_categorical(men_train, NUM_CLASSES)
    men_test = keras.utils.to_categorical(men_test, NUM_CLASSES)
    women_train = np.array(women_train, dtype=np.float64)
    women_test = np.array(women_test, dtype=np.float64)

    women_train /= 255
    women_test /= 255
    model.fit(women_train, men_train, batch_size = 1, epochs=200, verbose=1, validation_data=(women_test, men_test))
    score = model.evaluate(women_test, men_test, verbose=0)
    print('Потери на тесте:', score[0])
    print('Точность на тесте:', score[1])
    print("Baseline Error: %.2f%%" % (100 - score[1] * 100))
    model.save('myModelV3.h5') # Сохраняем в файл
    print("Модель сохранена")

def test_conv_model(model):
    cnt = 0
    ds_dir = "dataset/test/"
    filelist = os.listdir(ds_dir)
    for i in filelist:
        print(str(ds_dir+i))
        img = load_img_from_file(str(ds_dir + i), WORK_SIZE)
        img = np.array(img, dtype=np.float64)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)
        # Переводим числа к отрезку [0;1]
        img /= 255
        res = model.predict(img)
        if res[0][0] > res[0][1] and str(i).find('women') or res[0][1] > res[0][0] and str(i).find("men"):
            cnt += 1

    print(cnt / len(filelist) * 100)

def learn_saved_conv_model(model, filename):
    ds_dir = "dataset"
    women_train, men_train, women_test, men_test = load_dataset(ds_dir)
    # Приводим данные к нужному виду
    men_train = keras.utils.to_categorical(men_train, NUM_CLASSES)
    men_test = keras.utils.to_categorical(men_test, NUM_CLASSES)
    # В нужные массивы
    women_train = np.array(women_train, dtype=np.float64)
    women_test = np.array(women_test, dtype=np.float64)
    # Приводим к отрезку [0;1]
    women_train /= 255
    women_test /= 255

    tf.config.run_functions_eagerly(True)

    # Загрузка модели
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.fit(women_train, men_train, batch_size=50, epochs=200, verbose=1, validation_data=(women_test, men_test))
    score = model.evaluate(women_test, men_test, verbose=0)
    print('Потери на тесте:', score[0])
    print('Точность на тесте:', score[1])
    print("Baseline Error: %.2f%%" % (100 - score[1] * 100))
    model.save(filename)  # Сохраняем в файл
    print("Модель сохранена")


# Основная функция программы
if __name__ == '__main__':

    """Блок создания и обучения модели"""
   # model = create_conv_model()
   # learn_conv_model(model) # Обучаем модель

    """Блок тестирования/обучения уже созданной модели"""
    model = load_model('myModelV3.h5')
    print(model.summary())
    learn_saved_conv_model(model, "myModelV3.h5")

