import sys
import numpy as np
import copy

# import aima shit
sys.path.insert(1, './aima-python')
from search import *

class Monkey(Problem):
    
    def __init__(self, initial, goal, matrix: np.array):
        """ Define goal state and initialize a problem """
        self.initial = initial
        self.goal = goal

        Problem.__init__(self, initial, goal)

        self.rooms = matrix
        self.lenght = matrix.shape[0]

    def __str_index(self, x: int, y: int) -> str:
        return f"{x},{y}"

    def actions(self, state: dict) -> list:
        possible_actions = []
        
        # tupla(x, y). Coordinate della cella attuale del robot
        room_index = state["monkey"]
        
        # "x,y". Stringa per accedere alla relativa stanza nel dizionario
        room_str = self.__str_index(*room_index)
        
        if state[room_str] == 3. or state[room_str] == 21.:
            # definire azione di pulizia di una stanza
            possible_actions.append("CLEAN_"+room_str)

        
        # Controlla dove puo' muoversi
        # Controlla anche se la casella in cui vorra' muoversi e' un muro,
        # in caso lo evita
        
        # sinistra
        if room_index[1] > 0 and (state[self.__str_index(room_index[0], room_index[1]-1)] != 23.):
            possible_actions.append("MOVE_"+room_str+"_"+ f"{room_index[0]},{room_index[1]-1}") # si muove a sinistra
            
        # destra
        if room_index[1] < self.lenght - 1  and (state[self.__str_index(room_index[0], (room_index[1] + 1))] != 23.):
            possible_actions.append("MOVE_"+room_str+"_"+ f"{room_index[0]},{room_index[1] + 1}")
            
        # su
        if (room_index[0] > 0 ) and (state[self.__str_index(room_index[0]-1, room_index[1])] != 23.):
            possible_actions.append("MOVE_"+room_str+"_"+ f"{room_index[0]-1},{room_index[1]}")
            
        # giu
        if (room_index[0] < self.lenght - 1 ) and (state[self.__str_index(room_index[0]+1, room_index[1])] != 23.):
            possible_actions.append("MOVE_"+room_str+"_"+ f"{room_index[0]+1},{room_index[1]}")
        
        return possible_actions



    def result(self, state: dict, action: str) -> dict:
        """
        Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state
        """
        newstate = copy.deepcopy(state)
        act = action.split("_")

        # tupla(x, y). Coordinate della stanza nella matrice
        room_index = state["monkey"]

        # "x,y". Stringa per accedere alla relativa stanza nel dizionario
        room_str = self.__str_index(*room_index)
        
        # effettua l'azione
        if act[0] == 'CLEAN':
            if state[room_str] == 3.:
                newstate[room_str] = 2.
            elif state[room_str] == 21.:
                newstate[room_str] = 3.

        elif act[0] == "MOVE":

            next_room = act[2]            
            tmp =  next_room.split(',')
            newstate["monkey"] = (int(tmp[0]), int(tmp[1]))
        return newstate


    # Ritorna True se lo stato passatogli e' lo stato goal
    #
    # @state: stato attuale
    # @return: e' lo stato finale ?
    def goal_test(self, state: dict) -> bool:
        """ Given a state, return True if state is a goal state or False, otherwise """
        if state == self.goal:
            return True

        return False


    def astar_cost(self, node):
        return self.heuristic(node)+ node.path_cost
    

    def h(self, node):
        """ Return the heuristic value for a given state."""
        count=0
        
        for stanza in node.state:
            #caso in cui la stanza del nodo è diversa dallo stato goal
            if node.state[stanza] != self.goal[stanza]:
                if node.state[stanza] == 3.:
                    count+= 20
                count+= 1
        return count