import sys
import json
import copy
import numpy as np
from math import sqrt
from Animation import Animation
from Monkey import Monkey
from time import time

sys.path.insert(1, './aima-python')
from search import *
import os


'''
Lo stato iniziale e' rappresentato da un array con 2 elementi:
    [0]: Matrice che rappresenta lo spazio  (array multidimenzionale)
    [1]: La cella di partenza della scimmia (tupla)

Il tutto viene trasformato in JSON (!!!) per far funzionare AIMA-PYTHON

Lo stato finale e' identico a quello iniziale solo che le posizioni dell'array rappresentano:
    [0]: La Matrice che ci si aspetta di avere alla fine dell'esecuzione
    [1]: La cella finale dove dovra' essere posizionata la scimmia

Esempio di stato iniziale e finale
initial_state = [
    [[matrice con codice numerico di stanze di inizio]],
    (x_scimmia_iniziale,y_scimmia_iniziale)
    ]

goal_state = [
    [[matrice con codice numerico di stanze di fine]],
    (x_scimmia_finale, y_scimmia_finale)
    ]
'''

def getStatesFromMatrix(matrix: np.array) -> tuple:
    # dimenzione della matrice
    n = matrix.shape[0]

    # crea la matrice dello stato goal
    goal_matrix = np.zeros(n*n, dtype="float32").reshape(n, n)

    # posizione iniziale e finale della scimmia
    start = None
    stop = None

    # controlla tutta la matrice e nel mentre assegna alla 
    # matrice Goal i valori che dovra' avere alla fine dell'esecuzione
    # (le stanze sporche dovranno essere pulite)
    for i in range(n):
        for j in range(n):
            # vede se la cella rappresenta l'inizio
            if matrix[i][j] == 3.:
                start = (i, j)                       # salva la posizione iniziale della scimmia
                goal_matrix[i][j] = matrix[i][j]     # lascia invariata la cella di start
            
            # vede se la cella rappresenta la fine
            elif matrix[i][j] == 2.:
                stop = (i, j)                       # salva la posizione finale della scimmia
                goal_matrix[i][j] = matrix[i][j]    # lascia invariata la cella di stop
            
            # vede se le celle sono inaccessibili e le assegna alla matrice goal
            elif matrix[i][j] == 5.:
                goal_matrix[i][j] = matrix[i][j]

            # tutte le celle sporche vengono messe a pulite nella matrice goal   
            else:
                goal_matrix[i][j] = 0.

    # crea lo stato iniziale e finale
    # (rappresentato da un array con 2 posizioni)
    initial = [matrix.tolist(), start]
    goal = [goal_matrix.tolist(), stop]

    # ritorna gli stati sotto forma di JSON (!!!)
    return (json.dumps(initial), json.dumps(goal))


def pathFinder(matrice: np.array):

    # clear in base al sistema dove viene fatto partire il programma
    os.system('cls' if os.name == 'nt' else 'clear')

    n = matrice.shape[0] # la matrice deve essere nxn

    # deduce lo stato iniziale e finale dalla matrice
    initial_state, goal_state = getStatesFromMatrix(matrice)

    # crea il problema
    p_monkey = Monkey(initial_state, goal_state, n)

    # -------------- A* -------------- #
    print("A*\n")

    # salva l'ora dell'inizio della prova per
    # calcolare il tempo di esecuzione
    start_time = time()

    # risolve il problema usano la A*
    result = astar_search(p_monkey)

    # stampa le soluzione, i passi impegati ed il tempo impegato usando A*
    print(f"Execution Time: {(time()-start_time)}")
    print(f"Solution:\n {result.solution()}")
    print(f"Path Cost: {result.path_cost}")
    
    # fa partire l'animazione con la soluzione trovata da A*
    Animation(matrice, n, result.solution()).start()
    
    # -------------- --- -------------- #
    
    # -------------- BFS -------------- #
    print("BFS\n")

    # salva l'ora dell'inizio della prova per
    # calcolare il tempo di esecuzione
    start_time = time()
    
    # risolve il problema usano la BFS
    result = breadth_first_tree_search(p_monkey)

    # stampa le soluzione, i passi impegati ed il tempo impegato usando BFS
    print(f"Execution Time: {(time()-start_time)}\n")
    print(f"Solution:\n {result.solution()}\n")
    print(f"Path Cost: {result.path_cost}\n")
    

    # fa partire l'animazione con la soluzione trovata da BFS
    Animation(matrice, n, result.solution()).start()
    
    # -------------- --- -------------- #
