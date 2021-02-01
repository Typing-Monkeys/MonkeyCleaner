from imgUtils import allInOnePrepare
from knn import knn_classifier
from pathFinding import pathFinder
from sys import argv
from ann import annClassifier


# funzione per stampare gli argomenti necessari per far funzionare il programma
def help():
    print(
        '''
        Puoi utilizzare main.py nel seguente mondo:
            python main.py [argomenti]

        Gli argomenti accettati sono:

            --ann: per scegliere ANN come classificatore
            --knn: per scegliere KNN come classificatore
    
        ''')


# Il principio di tutto
# Se l'argomento passato è --knn allora fa una classificazione con il knn
# Se l'argomento è --ann gli va passato un secondo argomento che può essere di 2 tipi :
# --training -> fa il training del modello
# --testing -> classifica l'immagine di input con ann
def main(argv: list):
    if len(argv) > 0:
        if argv[0] == '--knn':
            
            # Prepara il dataset di training
            dati_testing = None

            if len(argv) == 2:
                dati_testing = allInOnePrepare(fromfile=True, fname=argv[1])
            else:
                dati_testing = allInOnePrepare()

            # classifica i dati con knn
            knn_result = knn_classifier(dati_testing)
            # Passa la matrice riconosciuta attraverso KNN agli algoritmi di path Finding
            pathFinder(knn_result)

        elif argv[0] == '--ann':
            
            # Invoca il classificatore con artificial neural network
            # gli passa come argomento l'opzione di training o di testing
            ann_result = None
            
            if len(argv) == 2:
                ann_result = annClassifier(fromfile=True, fname=argv[1])
            else:
                ann_result = annClassifier()

            # Passa la matrice riconosciuta attraverso ANN agli algoritmi di path Finding  
            pathFinder(ann_result)
        
        else:
            help()
    else:
        help()


if __name__ == "__main__":
    main(argv[1:])
