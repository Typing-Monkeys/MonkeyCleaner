import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
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
    # data_train, data_test, label_train, label_test = train_test_split(x, y, test_size=0.1, random_state=69)

    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    # return (data_train, data_test, label_train, label_test)

    return (x, None, y, None)


def prepare_test(file_name, bw=True):
    test = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if bw:
        # inverte il colore dell'immagine in quanto noi primati
        # scriviamo negro su bianco
        test = cv2.bitwise_not(test)
    
    test = cv2.resize(test, (28, 28), interpolation=cv2.INTER_AREA)

    test = test.flatten()
    test = np.array(test, dtype=np.float32)

    return np.array([test])


def prova(file):
    img = cv2.imread(file)
    
    # Convert the image to gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Performing OTSU threshold 
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected. 
    # A smaller value like (10, 10) will detect  
    # each word instead of a sentence. 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    
    # Appplying dilation on the threshold image 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours 
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    print(len(contours))
    img2 = img.copy()
    cordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cordinates.append((x,y,w,h))
        #bounding the images
        img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),1)

    cv2.imshow('SESSO', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def prova2(file):
    im1 = cv2.imread(file, 0)
    im = cv2.imread(file)

    ret, thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    dilated_value = cv2.dilate(thresh_value,kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cordinates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cordinates.append((x,y,w,h))
        #bounding the images
        
        im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)

    cv2.imshow('SESSO', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():

    # prova('./Dataset_Artista/Sesso_Staccato.png')
    prova2('./Dataset_Artista/Sesso_Duro.png')
    '''
    # prepara il dataset
    data_train, data_test, label_train, label_test = prepare_data('trimmedData.csv')

    # stampa il data e labels
    # print(data_train, label_train, '\n')

    # KNN
    print("Inizio il training ...")
    start_time = time.time()

    knn = cv2.ml.KNearest_create()
    knn.train(data_train, cv2.ml.ROW_SAMPLE, label_train)

    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    test_C = prepare_test('Dataset_Artista/X_Cazzodecane.png')

    
    print("Inizio il testing ...")
    start_time = time.time()

    # ret, result, neighbours, dist = knn.findNearest(data_test, k=1)
    ret, result, neighbours, dist = knn.findNearest(test_C, k=3)
    
    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))


    print(result)
    '''
    '''
    # Accuracy
    matches = result==label_test
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print( accuracy )
    '''
    '''
    print('Previsione', 'Valore Esatto')
    print(result, "D che sarebbe 3")
    print(ret)
    print(neighbours)
    print(dist)
    '''

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