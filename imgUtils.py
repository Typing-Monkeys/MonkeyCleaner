import numpy as np
import cv2
from math import sqrt


# funzione di debug per far stampare una sola immagine
# ed attendere un evento dell'utente
def printImage(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# esegue tutte le operazioni necessarie per rendere leggibili le immagini
def allInOnePrepare():
    # prende l'immagine dalla webcam
    im = monkeySee()
    
    # Trova coordinate e Gerarchie dell'immagine 
    im, contorni, gerarchia = monkeyDetect(im)

    # Estrae solo le coordinate dellettere in modo non ordinato
    coordinate = monkeyExtract(contorni, gerarchia)

    # ordina le coordinate delle lettere come nell'immagine
    coordinate_ordinate = monkeySort(coordinate)

    # prende le lettere dall'immagine originale
    letters = monkeyGetLetters(im, coordinate_ordinate)

    # prepara le lettere estratte per passarle al KNN
    ready_letters = monkeyPrepareLetters(letters)

    return ready_letters


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
        
        im, _, _ = monkeyDetect(img, inwebcam=True)
        # mostra la camera a video
        cv2.imshow('MonekyCam', im)

        # registra tasti premuti
        k = cv2.waitKey(1)
        
        # ESC pigiato
        if k%256 == 27:
            print("Premuto ESC, OOGA BOOGA...")
            if result is None:
                exit(1)
            else:
                break
                

        # SPACE pigiato
        elif k%256 == 32:
            # cattura il frame corrente e lo mostra a video
            cv2.imshow('Foto scimmiesca catturata', img)
            result = img

    cv2.destroyAllWindows()

    return result


# Prende un immagine (presunta tabella) e 
# ne estrae le celle
def monkeyDetect(img, inwebcam=False):
    # Converte l'immagine in GrayScale
    im1 = img.copy()
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # immagine usata per l'output
    im = img.copy()

    # Elimina i grigi e trasforma l'immagine in Bianco su Nero
    ret, thresh_value = cv2.threshold(im1, 150, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    # dilata l'immagine per facilitare l'identificazione dei contorni
    kernel = np.ones((15, 15), np.uint8)
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


    # ritorna l'immagine con i contorni evidenziati 
    # se chiamata dentro la webcam
    return_img = None

    if inwebcam:
        return_img = im
    else:
        return_img = im1

    # li ho fatti ritornare perche' a quanto pare non si puo' stampare roba dentro a questa 
    # funzione perche opencv dice NO
    return (return_img, cordinates, hierarchy)


# seleziona solo le coordinate con la giusta Gerarchia
# (estra le coordinate solo delle lettere e non del resto)
def monkeyExtract(contorni, gerarchia):
    
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


# estrae le immagini dalla tebella
def monkeyGetLetters(img, contorni_ordinati):
    result = []

    for elem in contorni_ordinati:
        x, y, w, h = elem
        
        tmp = img[y:y+h, x:x+w]

        result.append(tmp)

    return result


# prepara le immagini estratte per passarle al KNN
def monkeyPrepareLetters(lettere_ordinate, bw=True):
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
