import time
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from math import sqrt


# Legge e prepara il Dataset per il Training
def prepare_data(file_name:str, split=False, t_size=0.1) -> tuple:
    print("Importo il dataset ...")
    start_time = time.time()

    # importa i ldataset con le label
    dataset = pd.read_csv(file_name, header=None).astype('float32')

    # elimino la prima colonna e tengo tutto il resto
    x = dataset.drop([0], axis=1).to_numpy()

    # estraggo solo la prima colonna
    y = dataset[[0]].to_numpy()
    
    if split:   
        # divide e mescola il dataset in training e testing
        data_train, data_test, label_train, label_test = train_test_split(x, y, test_size=t_size, random_state=69)
        
        print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

        return (data_train, data_test, label_train, label_test)

    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    # return (data_train, data_test, label_train, label_test)

    return (x, None, y, None)


def knn_classifier(dati_testing: np.array, k=3, toMatrix=True) -> np.array:
    data_train, data_test, label_train, label_test = prepare_data('trimmedData.csv')

    # KNN
    print("Inizio il training ...")
    start_time = time.time()

    knn = cv2.ml.KNearest_create()
    knn.train(data_train, cv2.ml.ROW_SAMPLE, label_train)
    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    print("Inizio il testing ...")
    start_time = time.time()

    # ret, result, neighbours, dist = knn.findNearest(data_test, k=1)
    ret, result, neighbours, dist = knn.findNearest(dati_testing, k=k)
    
    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    if toMatrix:
        result = result.reshape(
                int(sqrt(len(result))), 
                int(sqrt(len(result)))
            )
    else:
        result = result.flatten()

    return result
