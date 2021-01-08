import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


def prepare_data(file_name:str) -> dict:
    print("Importo il dataset ...")
    start_time = time.time()

    # importa i ldataset con le label
    dataset = pd.read_csv(file_name, header=None).astype('float32')

    # elimino la prima colonna e tengo tutto il resto
    x = dataset.drop([0], axis=1).to_numpy()

    # estraggo solo la prima colonna
    y = dataset[[0]].to_numpy()
    
    # divide e mescola il dataset in training e testing
    data_train, data_test, label_train, label_test = train_test_split(x, y, test_size=0.2, random_state=69)

    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    return (data_train, data_test, label_train, label_test)


def prepare_test(file_name):
    test = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    test = test.flatten()
    test = np.array(test, dtype=np.float32)
    print(test)


def main():

    '''
    prepare_test('Dataset_Artista/A.png')

    return
    '''
    
    # prepara il dataset
    data_train, data_test, label_train, label_test = prepare_data('trimmedData.csv')

    # stampa il data e labels
    print(data_train, label_train, '\n')

    # KNN
    print("Inizio il training ...")
    start_time = time.time()

    knn = cv2.ml.KNearest_create()
    knn.train(data_train, cv2.ml.ROW_SAMPLE, label_train)

    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    ret, result, neighbours, dist = knn.findNearest(np.array([data_test[0]]), k=3)

    print('Previsione', 'Valore Esatto')
    print(result, label_test[0])

if __name__ == "__main__":
    main()

'''
digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cells = np.hsplit(row, 50)
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype=np.float32)

print(cells)

k = np.arange(10)
cells_labels = np.repeat(k, 250)


test_digits = np.vsplit(test_digits, 50)
test_cells = []
for d in test_digits:
    d = d.flatten()
    test_cells.append(d)
test_cells = np.array(test_cells, dtype=np.float32)


# KNN
knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)


#print(result)
'''