import sys
import numpy as np
import copy
import json

sys.path.insert(1, './aima-python')
from search import *


class Monkey(Problem):
    def __init__(self, initial: json , goal: json, lenght: np.array):
        super().__init__(initial, goal)

        self.initial = initial
        self.goal = goal
        self.lenght = lenght

    # funzione che dati 2 interi restituisce una stringa che
    # verra' inserita nel nome del comando
    def __str_index(self, x: int, y: int) -> str:
        return f"{x},{y}"
    
    # data la posizione della scimmia guarda le possibili azioni che può compiere e le salva in una lista
    def actions(self, state: json) -> list:
        # trasforma il JSON (!!!) in un array usabile
        jstate = json.loads(state)

        # lista che contiene tutte le possibili azioni
        possible_actions = []
        
        # prende la posizione della scimmia
        x, y = jstate[1]

        # prende la matrice dello stato attuale
        matrice = jstate[0]
        
        # prende la stringa che andra' all'interno delle azioni
        cell_str = self.__str_index(x, y)


        # controlla se la cella in cui e' va pulita
        if matrice[x][y] == 1. or matrice[x][y] == 4.:
            possible_actions.append("CLEAN_"+cell_str)


        # Controlla dove puo' muoversi
        # Controlla anche se la casella in cui vorrà muoversi è una cella inaccessibile,
        # in caso la evita
        
        # Sinistra
        if y > 0 and (matrice[x][(y-1)] != 5.):
            possible_actions.append("MOVE_"+cell_str+"_"+ f"{x},{y-1}")

        # Destra
        if y < self.lenght - 1  and (matrice[x][(y + 1)] != 5.):
            possible_actions.append("MOVE_"+cell_str+"_"+ f"{x},{y + 1}")

        # Su
        if (x > 0 ) and (matrice[(x - 1)][y] != 5.):
            possible_actions.append("MOVE_"+cell_str+"_"+ f"{x-1},{y}")

        # Giu
        if (x < self.lenght - 1 ) and (matrice[(x+1)][y] != 5.):
            possible_actions.append("MOVE_"+cell_str+"_"+ f"{x+1},{y}")
        
        # ritorna la lista delle azioni possibili
        return possible_actions

    def result(self, state: json, action: str) -> json:
        # trasfroma il JSON (!!!) in un array usabile
        jstate = json.loads(state)

        # copia il vecchio stato per creare quello nuovo
        newstate = copy.deepcopy(state)

        # converte il nuovo stato (JSON [!!!]) in un array usabile
        newjstate = json.loads(newstate)

        # splitta le azioni per _
        # CLEAN_CELLA 
        # oppure
        # MOVE_CELLA_CELLA
        act = action.split("_")

        # posizione attuale della scimmia
        x, y = jstate[1]

        # matrice che rappresenta lo stato attuale del mondo
        matrice = jstate[0]

        # nuova matrice che rappresenta lo stato dopo aver
        # effettuato l'azione scelta
        newMatrice = newjstate[0]

        # pulisce la cella
        if act[0] == 'CLEAN':
            if matrice[x][y] == 1.:
                newMatrice[x][y] = 0.   # aggiorna la nuova matrice
            elif matrice[x][y] == 4.:
                newMatrice[x][y] = 1.

        # si muove sulla cella scelta
        elif act[0] == 'MOVE':
            # la cella in cui deve muoversi
            next_cell = act[2]    
            # dalla stringa prende le coordinate della cella in cui deve muoversi  
            tmp =  next_cell.split(',')
            
            # aggiorna le coordinate della scimmia
            newjstate[1] = (int(tmp[0]), int(tmp[1]))

        # ritorna il nuovo stato sotto forma di JSON (!!!)
        return json.dumps(newjstate)

    # Ritorna True se lo stato passatogli e' lo stato goal
    #
    # @state: stato attuale
    # @return: e' lo stato finale ?
    def goal_test(self, state: json) -> bool:
        # trasforma gli stati iniziali e finali da JSON (!!!)
        # in array utilizzabili
        jstate = json.loads(state)
        jgoal = json.loads(self.goal)

        # prende la matrice e la posizione attuale della scimmia
        # dallo stato attuale
        matrice, monkey = jstate

        # trasforma la matrice in un np.array per
        # sfruttare le sue funzioni belle di comparazione :^)
        matrice = np.array(matrice)

        # converte la matrice goal in un np.array
        # per sfruttare le sue funzioni belle di comparazione :^)
        goal = np.array(jgoal[0])

        # controlla se tutti gli elementi delle matrici sono uguali
        result = matrice == goal

        # se i 2 stati corrispondono allora siamo nello stato Goal
        if result.all() and monkey == jgoal[1]:
            return True

        # non siamo nello stato Goal
        return False

    # MissPlacedBananasHeuristica per A*
    def h(self, node):
        # trasforma gli stati iniziali e finali da JSON (!!!)
        # in array utilizzabili
        jstate = json.loads(node.state)
        jgoal = json.loads(self.goal)

        # converte le matrici degli stati in np.array
        # per sfruttare le sue funzioni belle di comparazione :^)
        matrice = np.array(jstate[0])
        goal = np.array(jgoal[0])

        # controlla quanti elementi sono uguali
        cmp = matrice == goal

        # ritorna il numero di elementi differenti
        return np.count_nonzero(cmp == False)
