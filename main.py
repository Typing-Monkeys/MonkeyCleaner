import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from math import sqrt
import operator

# Legge e prepara il Dataset per il Training
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


# Legge un'immagine da file e la prepara per essere usata come elemento per il Testing
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


def monkeyPrepareTest(lettere_ordinate, bw=True):
    
    result = []
    
    for elem in lettere_ordinate:
        # inverete i colori di tutte le immagini
        tmp = cv2.bitwise_not(elem)
        tmp = cv2.resize(tmp, (28, 28), interpolation=cv2.INTER_AREA)
        tmp = tmp.flatten()        

        tmp = np.array(tmp, dtype=np.float32)
  
        result.append(tmp)

    return np.array(result)


# Prende un immagine (presunta tabella) da file e 
# ne estrae le celle
def monkeyRead(file, print=False):
    # Importa 2 volte l'immagine per modificarne una
    im1 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    ret, thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((20, 20), np.uint8)
    dilated_value = cv2.dilate(thresh_value,kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cordinates.append((x, y, w, h))

        # diesegna un rettangolo sopra ogni elemento trovato
        im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)

    # stampa l'immagine con i rettangoli
    # cv2.imshow('Immagine Inscatolata', im)
    
    # li ho fatti ritornare perche' a quanto pare non si puo' stampare roba dentro a questa 
    # funzione perche opencv dice NO
    return (im, cordinates, hierarchy)
    

def monkeyGetImages(img, contorni_ordinati):
    result = []

    for elem in contorni_ordinati:
        x, y, w, h = elem
        
        tmp = img[y:y+h, x:x+w]

        result.append(tmp)

    return result


def monkeyExtract(contorni, gerarchia):
    gerarchia_size = len(gerarchia[0])
    
    g_temp = gerarchia[0]
    # print(gerarchia_size)
    result = []
    
    i = 0
    while True:
        if g_temp[i][-1] == -1:
            if g_temp[i][0] == -1:
                break
            else:
                result.append(contorni[i])
        i += 1

    return result


def monkeySort(cordinate):
    def miniSort(cordinate, index=0):
        cordinate.sort(key = lambda x: x[index])
        
        return cordinate

    miniSort(cordinate, index=1)

    row_size = sqrt(len(cordinate))

    # controlla se la matirce e' quadrata:
    # se non e' quadrata ritorna un errere, altrimenti
    # converte in intero row_size
    if row_size % 1 != 0.:
        print("Non e' una matrice quadrata !")
        exit(1)
    else:
        row_size = int(row_size)

    i = 0
    start = 0
    stop = row_size
    result = []

    while i < row_size:
        result += miniSort(cordinate[start:stop], index=0)

        start = stop
        stop += row_size
        i += 1

    
    return result


def main():
    # legge immagine di test
    im, contorni, gerarchia = monkeyRead('./Dataset_Artista/Esempio3.png')

    # estrae le lettere in modo non ordinato
    lista = monkeyExtract(contorni, gerarchia)
    # ordina le lettere come nell'immagine
    lista = monkeySort(lista)

    # prende le lettere dall'immagine originale
    lista = monkeyGetImages(im, lista)

    # print(np.array(lista))

    # prepara i dati per passarli al KNN
    dati_testing = monkeyPrepareTest(lista)

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

    print("Inizio il testing ...")
    start_time = time.time()

    # ret, result, neighbours, dist = knn.findNearest(data_test, k=1)
    ret, result, neighbours, dist = knn.findNearest(dati_testing, k=3)
    
    print("Fatto: --- %s seconds ---\n" % round(time.time() - start_time, 2))

    result = result.reshape(
            int(sqrt(len(result))), 
            int(sqrt(len(result)))
        )

    print(result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
