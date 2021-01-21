from sys import argv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from emnist import extract_training_samples
from main import prepare_data
from main import printImage
from keras import optimizers
from keras.models import load_model
from main import monkeyPrepareTest


def training():
    # Crea un modello di Rete Neurale
    def makeANNModel1():
        model  = keras.Sequential()

        model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))
        model.summary()
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        return model

    # importa il dataset per il training
    x_train, x_test, y_train, y_test = prepare_data('trimmedData.csv', split=True, t_size=0.1)

    # normalizza i valori
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # cambia le lable del TESTING per renderle sequenziali
    # 0, 1, ...

    n_y_test = []
    
    for elem in y_test:
        if elem == 2.:
            n_y_test.append(0.)
        elif elem == 3.:
            n_y_test.append(1.)
        elif elem == 5.:
            n_y_test.append(2.)
        elif elem == 18.:
            n_y_test.append(3.)
        elif elem == 21.:
            n_y_test.append(4.)
        elif elem == 23.:
            n_y_test.append(5.)

    n_y_test = np.array(n_y_test, dtype="float32")

    # cambia le lable del TRAINING per renderle sequenziali
    # 0, 1, ...
    n_y_train = []
    
    for elem in y_train:
        if elem == 2.:
            n_y_train.append(0.)
        elif elem == 3.:
            n_y_train.append(1.)
        elif elem == 5.:
            n_y_train.append(2.)
        elif elem == 18.:
            n_y_train.append(3.)
        elif elem == 21.:
            n_y_train.append(4.)
        elif elem == 23.:
            n_y_train.append(5.)

    n_y_train = np.array(n_y_train, dtype="float32")

    # trasforma le lable in np.array
    y_test = n_y_test
    y_train = n_y_train

    # inizializza il modello scelto
    model = makeANNModel1()

    # inizializza le epoche e i batch
    b_size = 128
    epoche = 16
    
    # allena il modello
    history = model.fit(x_train, y_train, batch_size=b_size, epochs=epoche, validation_data=(x_test, y_test))

    # esporta il modello
    model.save(f'./Models/model_b{b_size}_e{epoche}.h5')


def useModel(model_name, test_images: list, test_labels=None):
    # Restore the weights
    model = load_model(model_name)

    # res = model.predict_classes(test_images[0:1])
    # print(res, test_labels[0])
    for elem in test_images:
        res = model.predict_classes(test_images)
        print(res)
    

def main(argv):
    if argv[0] == '--training':
        training()

    elif argv[0] == '--testing':
        image_test = cv2.imread('Dataset_Artista/F_Nero.png', 0)

        test = monkeyPrepareTest([image_test])
        
        useModel("./Models/model_b128_e16.h5", test)


if __name__ == "__main__":
    main(argv[1:])