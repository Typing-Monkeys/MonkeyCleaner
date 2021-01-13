import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from math import sqrt
import operator
import threading


def printImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# fa scegliere all'utente il frame da prendere 
# dalla webcam
def monkeySee(mirror=False):
    # apre lo stream della camera
    cam = cv2.VideoCapture(0)

    # conterra' l'immagine finale
    result = None

    while True:
        # legge frame by frame
        _, img = cam.read()

        # specchia l'immagine se richiesto
        if mirror: 
            img = cv2.flip(img, 1)
            
        # mostra la camera a video
        cv2.imshow('MonekyCam', img)

        # registra tasti premuti
        k = cv2.waitKey(1)
        
        # ESC pigiato
        if k%256 == 27:
            print("Premuto ESC, OOGA BOOGA...")
            break

        # SPACE pigiato
        elif k%256 == 32:
            # cattura il frame corrente e lo mostra a video
            cv2.imshow('Frame catturato', img)
            result = img

    cv2.destroyAllWindows()

    return result


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


# prepara le immagini estratte per passarle al KNN
def monkeyPrepareTest(lettere_ordinate, bw=True):
    result = []
    
    for elem in lettere_ordinate:

        # Crea contrasto tra bianchi e neri trasformando l'immagine in Bianco su Nero
        _, threshold = cv2.threshold(elem,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #tmp = cv2.bitwise_not(threshold)

        # Rende l'immagine di dimenzioni 28x28
        tmp = cv2.resize(threshold, (28, 28), interpolation=cv2.INTER_AREA)

        # spalma l'immagine su un array monodimenzionale
        tmp = tmp.flatten()        

        # converte l'array in un numpy array
        tmp = np.array(tmp, dtype=np.float32)
  
        result.append(tmp)

    return np.array(result)


# Prende un immagine (presunta tabella) e 
# ne estrae le celle
def monkeyDetect(img, print=False):
    # Converte l'immagine in GrayScale
    im1 = img.copy()
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # immagine usata per l'output
    im = img.copy()

    # Elimina i grigi e trasforma l'immagine in Bianco su Nero
    ret, thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # dilata l'immagine per facilitare l'identificazione dei contorni
    kernel = np.ones((20, 20), np.uint8)
    dilated_value = cv2.dilate(thresh_value,kernel, iterations=1)

    # trova i contorni nell'immagine
    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # aggiunge le coordinate all'array da ritornare
    cordinates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cordinates.append((x, y, w, h))

        # diesegna un rettangolo sopra ogni elemento trovato
        im = cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)

    # Stampa l'immagine originale con evidenziato i bordi trovati
    printImage(im, "Immagine con Bordi")

    # li ho fatti ritornare perche' a quanto pare non si puo' stampare roba dentro a questa 
    # funzione perche opencv dice NO
    return (im1, cordinates, hierarchy)
    

# estrae le immagini dalla tebella
def monkeyGetImages(img, contorni_ordinati):
    result = []

    for elem in contorni_ordinati:
        x, y, w, h = elem
        
        tmp = img[y:y+h, x:x+w]

        result.append(tmp)

    return result


# seleziona solo le coordinate con la giusta Gerarchia
# (estra le coordinate solo delle lettere e non del resto)
def monkeyExtract(contorni, gerarchia):
    gerarchia_size = len(gerarchia[0])
    
    g_temp = gerarchia[0]

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


# ordina le coordinate in base alla loro posizione sul foglio
# per farle corrispondere alla matrice originale
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
    # prende l'immagine dalla webcam
    im = monkeySee()

    # Trova coordinate e Gerarchie dell'immagine 
    im, contorni, gerarchia = monkeyDetect(im)

    # Estrae solo le coordinate dellettere in modo non ordinato
    lista = monkeyExtract(contorni, gerarchia)

    # ordina le coordinate delle lettere come nell'immagine
    lista = monkeySort(lista)

    # prende le lettere dall'immagine originale
    lista = monkeyGetImages(im, lista)

    # prepara le lettere estratte per passarle al KNN
    dati_testing = monkeyPrepareTest(lista)

    # prepara il dataset
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
