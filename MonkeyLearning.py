from sys import argv
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from knn import prepare_data
from keras import optimizers
from keras.models import load_model
from imgUtils import monkeyPrepareLetters


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

        test = monkeyPrepareLetters([image_test])
        
        useModel("./Models/model_b128_e16.h5", test)


if __name__ == "__main__":
    main(argv[1:])