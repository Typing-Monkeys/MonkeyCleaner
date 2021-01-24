from ann import useModel
from emnist import extract_training_samples
from imgUtils import printImage
import numpy as np
import os

lettere_nostre = [2, 3, 5, 18, 21, 23]

nuoveLables = {
    '2': '0',
    '3': '1',
    '5': '2',
    '18': '3',
    '21': '4',
    '23': '5'
    }
  

def main():


    images, labels = extract_training_samples('letters')
    labels = labels - 1

    newimgs = []
    newlables = []

    for i in range(len(labels)):
        tmp = labels[i]

        if tmp in lettere_nostre:
            newimgs.append(images[i].flatten())
            newlables.append(tmp)
    
    tmpLable = []
    for elem in newlables:
        tmp = nuoveLables[str(elem)]

        tmpLable.append(int(tmp))

    newlables = np.array(tmpLable)

    test = np.array(newimgs)

    models = os.listdir('./Models')

    risultati = []

    for elem in models:
        result = useModel(f"./Models/{elem}", test, toMatrix=False)

        '''
        print(result)
        print(newlables)
        '''
        
        cmp = result == newlables
        zeros = np.count_nonzero(cmp == False)
        
        accuracy = 1 - (zeros/len(newlables))
        
        '''
        print('Results Len: ')
        print(len(result))
        print('\nAccuracy: ')
        print(accuracy)
        '''
        risultati.append((elem, accuracy))

    os.system('cls' if os.name == 'nt' else 'clear')
    
    risultati.sort(key=lambda x: x[1])

    for elem in risultati:
        print(f"Model Name: {elem[0]}")
        print(f"Accuracy: {elem[1]}\n")

if __name__ == "__main__":
    main()