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


def useModel(model_name, test_images: np.array, test_labels=None):
    # Carica un modello già allenato
    model = load_model(model_name)
    # Riconosce le lettere e le ritorna sottoforma di matrice di Float
    res = model.predict_classes(test_images)

    res = res.reshape(
            int(sqrt(len(res))), 
            int(sqrt(len(res)))
        ).astype("float32")

    return res


def annClassifier(argomento):
    # Inizia il training con bachsize e epoche di default
    if argomento == '--training':
        training()
        exit()
    # Fa training ripetuto con diversi bachsize e epoche e ogni volta ne salva il modello creato 
    #(funzione principalmente per il testing di diversi modelli di ANN)
    elif argomento == '--Mtraining':
        # i dati sono nel seguente formato:
        # (batch_size, epoche)
        t_data = [
            (128, 16), #default
            (10, 16), # probabile overfitting
            (5000, 500), # ?
            (64, 10), # non ci aspettiamo tanti problemi
            ]
        
        for elem in t_data:
            training(*elem)

        exit()
    
    # Se l'arogmento passato al main è --testing usa un modello già allenato per effettuare la classificazione
    # Ritorna un mumpyArray che verrà usato dal pathfinding
    elif argomento == '--testing':
        image_test = allInOnePrepare()
        return useModel("./Models/model_b128_e16.h5", image_test)

    # Contorllo degli argomenti passati, se sono errati ritorna l'avviso con il comando errato passatogli
    elif argomento != '--training' and argomento != '--testing' and argomento != '--Mtraining':
        print(f"Errore nel parametro di input, non risconosciuto: {argomento}")
        exit()