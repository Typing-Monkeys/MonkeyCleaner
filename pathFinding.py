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
        "monkey":
            {
                "room": (1, 1) # la cella con valore S (18.)
            },
        "rooms":
            {
                # sono gli indici di tutte le stanze
                "0,0": 
                    {
                        "state": "dirty"
                    },
                "0,1": 
                    {
                        "state":"clean"
                    },
                "0,2": 
                    {
                        "state":"inaccessible"
                    },
                "1,0": 
                    {
                        "state":"very_dirty"
                    },
                "1,1": 
                    {
                        "state":"start"
                    },
                "1,2": 
                    {
                        "state":"finish"
                    },
                "2,0": 
                    {
                        "state":"clean"
                    },
                "2,1": 
                    {
                        "state":"clean"
                    },
                "2,2": 
                    {
                        "state":"clean"
                    },
                "3,0": 
                    {
                        "state":"clean"
                    }, 
                "3,1": 
                    {
                        "state":"clean"
                    },
                "3,2": 
                    {
                        "state":"clean"
                    }
            }
    }
'''

def matrixToDict(matrix):
    initial = {
        "monkey": {},
        "rooms": {}
    }


    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]

            if val == 18.:
                initial["monkey"] = {"room": (i, j)}
            
            initial["rooms"][f'{i},{j}'] = {"state": matrix[i][j]}

    goal = copy.deepcopy(initial)
    
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]

            if val == 5. or val == 18. or val == 23.:
                if val == 5.:
                    goal["monkey"] = {"room": (i, j)}

            else:
                goal["rooms"][f'{i},{j}'] = {"state": 2.}
    
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

    print()
    print(result.solution())
    print()
    print(result.state)
    print(result.path_cost)
    print(f"{(time()-start_time)}")

    Animation(matrice, 3, 200).start()


if __name__ == "__main__":
    main()