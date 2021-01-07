import cv2
import numpy as np
import pandas as pd
import time


def prepare_data(file_name:str) -> tuple:
    print("Importo il dataset ...")
    start_time = time.time()
    # importa i ldataset con le label
    dataset = pd.read_csv(file_name).astype('float32')
    dataset.rename(columns={'0':'label'}, inplace=True)

    # Splitta il dataset la X - Nostri dati , e y - Label dei predict
    x = dataset.drop('label', axis=1).to_numpy()
    y = dataset['label'].to_numpy()

    print("Fatto: --- %s seconds ---" % round(time.time() - start_time, 2))

    # print("OOKKEEY")

    # x: dati
    # y: label
    return (x, y)


def prepare_test(file_name):
    test = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    test = test.flatten()
    test = np.array(test, dtype=np.float32)
    print(test)


def main():

    prepare_test('A.png')

    return

    dati, labels = prepare_data('A_Z Handwritten Data.csv')

    print(dati, labels)

    # KNN
    print("Inizio il training ...")
    start_time = time.time()

    knn = cv2.ml.KNearest_create()
    knn.train(dati, cv2.ml.ROW_SAMPLE, labels)

    print("Fatto: --- %s seconds ---" % round(time.time() - start_time, 2))

    # ret, result, neighbours, dist = knn.findNearest(test_cells, k=3)

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