from sys import argv
import numpy as np
import cv2
from math import sqrt
from tensorflow import keras
from tensorflow.keras import layers
from knn import prepare_data
from keras import optimizers
from keras.models import load_model
from imgUtils import allInOnePrepare


def training(b_size=128, epoche=16):
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
    # b_size = 128
    # epoche = 16
    
    # allena il modello
    history = model.fit(x_train, y_train, batch_size=b_size, epochs=epoche, validation_data=(x_test, y_test))

    # esporta il modello
    model.save(f'./Models/model_b{b_size}_e{epoche}.h5')


def useModel(model_name, test_images: np.array, test_labels=None, toMatrix=True):
    # Carica un modello già allenato
    model = load_model(model_name)
    # Riconosce le lettere e le ritorna sottoforma di matrice di Float
    res = model.predict_classes(test_images)

    if toMatrix:
        res = res.reshape(
                int(sqrt(len(res))), 
                int(sqrt(len(res)))
            ).astype("float32")

    return res

def annClassifier(fromfile=False, fname='test'):
    # Se l'arogmento passato al main è --testing usa un modello già allenato per effettuare la classificazione
    # Ritorna un mumpyArray che verrà usato dal pathfinding    
    image_test = None

    if fromfile == True:
        image_test = allInOnePrepare(fromfile=fromfile, fname=fname)
    else:
        image_test = allInOnePrepare()
    
    return useModel("./Models/model_b10_e16.h5", image_test)
