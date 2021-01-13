import numpy as np
from math import sqrt
import sys
import copy
from Animation import Animation
from Monkey import Monkey
from time import time
# import aima shit
sys.path.insert(1, './aima-python')
from search import *

'''
initial_state = {
                "monkey": (1, 1) # la cella con valore S (18.),
                
                "0,0": 3.
                "0,1": 18.
                "0,2": 2.
                "1,0": 3.
                "1,1": 23.
                "1,2": Robe...
                "2,0": 
                "2,1": 
                "2,2": 
                "3,0": 
                "3,1": 
                "3,2": 
                    
            }
    }
'''

def matrixToDict(matrix):
    initial = {}


    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]

            if val == 18.:
                initial["monkey"] = (i, j)
            
            initial[f'{i},{j}'] = matrix[i][j]

    goal = copy.deepcopy(initial)
    
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]

            if val == 5. or val == 18. or val == 23.:
                if val == 5.:
                    goal["monkey"] = (i, j)

            else:
                goal[f'{i},{j}'] = 2.
    
    return (initial, goal)



def main():
    a = [3., 3., 5., 23., 21., 23., 2., 2., 18.]
    matrice = np.array(a).reshape(int(sqrt(len(a))), int(sqrt(len(a))))

    # Animation(matrice, 3, 200).start()

    initial_state, goal_state = matrixToDict(matrice)

    print(initial_state, goal_state)

    p_monkey = Monkey(initial_state, goal_state, matrice)

    start_time = time()
    result = breadth_first_tree_search(p_monkey)

    print("BFS")
    print()
    print(result.solution())
    print()
    print(result.state)
    print(result.path_cost)
    print(f"{(time()-start_time)}\n")

    Animation(matrice, 3, 200).start()

    start_time = time()
    result = astar_search(p_monkey)

    print("A*")
    print()
    print(result.solution())
    print()
    print(result.state)
    print(result.path_cost)
    print(f"{(time()-start_time)}")

    Animation(matrice, 3, 200).start()


if __name__ == "__main__":
    main()