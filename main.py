from imgUtils import allInOnePrepare
from knn import knn_classifier

def main():
    
    dati_testing = allInOnePrepare()

    knn_result = knn_classifier(dati_testing)

    print(knn_result)

if __name__ == "__main__":
    main()
