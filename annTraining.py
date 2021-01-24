from sys import argv
from ann import training


def main(argvs):
    if len(argvs) > 0:
        # Fa training ripetuto con diversi bachsize e epoche e ogni volta ne salva il modello creato 
        #(funzione principalmente per il testing di diversi modelli di ANN)
        if argvs[0] == '--Mtraining':
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

        elif len(argvs) == 2:
            # prende i batch_size e le epoche dagli argomenti passati
            b = int(argvs[0])
            e = int(argvs[1])
            
            training(b_size=b, epoche=e)
    else:        
        # Inizia il training con bachsize e epoche di default
        training()


if __name__ == "__main__":
    main(argv[1:])