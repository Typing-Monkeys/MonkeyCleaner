from imgUtils import allInOnePrepare
from knn import knn_classifier
from pathFinding import pathFinder
from sys import argv
from MonkeyLearning import annClassifier

# Il principio di tutto
# Se l'argomento passato è --knn allora fa una classificazione con il knn
# Se l'argomento è --ann gli va passato un secondo argomento che può essere di 2 tipi :
# --training -> fa il training del modello
# --testing -> classifica l'immagine di input con ann
def main(argv):

    if argv[0] == '--knn':
        
        # Prepara il dataset di training
        dati_testing = allInOnePrepare()
        # classifica i dati con knn
        knn_result = knn_classifier(dati_testing)
        # Passa la matrice riconosciuta attraverso KNN agli algoritmi di path Finding
        pathFinder(knn_result)

    elif argv[0] == '--ann':

        #Controllo il giusto numero di argomenti passatagli
        if len(argv) < 2:

            print("Passami --training oppure --testing")
            return
        
        # Invoca il classificatore con artificial neural network
        # gli passa come argomento l'opzione di training o di testing
        ann_result = annClassifier(argv[1])
        # Passa la matrice riconosciuta attraverso ANN agli algoritmi di path Finding  
        pathFinder(ann_result)
        

if __name__ == "__main__":
    main(argv[1:])
