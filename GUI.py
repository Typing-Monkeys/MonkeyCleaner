import numpy as np
import pygame
from math import sqrt
from time import sleep


class Animation:
    def __init__(self, matrix: np.array, dimension: int, solutions: list, cellsize=50, title="Peppone's Bizarre Adventure"):
        # la matrice viene copiata per evitare che venga modificata anche al di fuore di questa classe
        self.matrix = np.copy(matrix)
        
        # dimensioni della stampa a video
        self.dimension = dimension
        self.cellsize = cellsize

        self.solutions = solutions

        # titolo della finestra
        self.title = title

        self.col_grid = (255, 255, 255)
        self.surface = None        

    def __load_imgs(self):
        self.startbg = pygame.image.load('./Animation_imgs/Appartamento_Scandinavo_di_Peppone.png').convert_alpha()
        self.startbg = pygame.transform.scale(self.startbg, (self.cellsize, self.cellsize))

        self.stopbg = pygame.image.load('./Animation_imgs/Letto_Scandinavo_di_Peppone.png').convert_alpha()
        self.stopbg = pygame.transform.scale(self.stopbg, (self.cellsize, self.cellsize))

        self.banana = pygame.image.load('./Animation_imgs/Banana.png').convert_alpha()
        self.banana = pygame.transform.scale(self.banana, (self.cellsize, self.cellsize))
        
        self.bananas = pygame.image.load('./Animation_imgs/Banana_Doppia.png').convert_alpha()
        self.bananas = pygame.transform.scale(self.bananas, (self.cellsize, self.cellsize))

        self.wall = pygame.image.load('./Animation_imgs/Foresta.png').convert_alpha()
        self.wall = pygame.transform.scale(self.wall, (self.cellsize, self.cellsize))

        self.pratino = pygame.image.load('./Animation_imgs/Pratino.png').convert_alpha()
        self.pratino = pygame.transform.scale(self.pratino, (self.cellsize, self.cellsize))

        self.peppone = pygame.image.load('./Animation_imgs/Peppone.png').convert_alpha()
        self.peppone = pygame.transform.scale(self.peppone, (self.cellsize, self.cellsize))

        self.walk_peppone = pygame.image.load('./Animation_imgs/Walking_Peppone.png').convert_alpha()
        self.walk_peppone = pygame.transform.scale(self.walk_peppone, (self.cellsize, self.cellsize))

        self.flip_walk_peppone = pygame.transform.flip(self.walk_peppone, True, False)

        # self.rect = self.startbg.get_rect()

        # contiene le relazioni Numero-Lettera e Lettera-Numer
        # Nelle posizioni del dizionario rappresentate da numeri ('2.0') ci sono il nome della corrispettiva 
        # lettera e il colore che la rappresenta.
        # 
        # Nelle posizione rappresentate dalle lettere ('c') ci sono i corrispettivi numeri utilizzati
        # per codificare le lettere. Queste ultime posizioni sono solo per creare la matrice a mano, in quanto
        # nel problema reale avremmo una matrice gia' popolata da numeri e non andra' creata.
        self.database = {
            '0.0': self.pratino,

            '1.0': self.banana,

            '2.0': self.stopbg,

            '3.0': self.startbg,

            '4.0': self.bananas,

            '5.0': self.wall,

            'c': 0.,
            'd': 1.,
            'f': 2.,
            's': 3.,
            'v': 4.,
            'x': 5.
        }

    def __print_GUI(self):
        pygame.mixer.music.load('./Animation_imgs/Monkey_beat.mp3')
        pygame.mixer.music.play(-1, 0.0)
        pygame.mixer.music.set_volume(0.1)

        '''
        soundObj = pygame.mixer.Sound('./Animation_imgs/Monkey_beat.mp3')
        soundObj.play()
        '''

        for i in range(self.dimension):
            for j in range(self.dimension):
                # colora la cella in base al suo contenuto
                dict_index = str(self.matrix[i][j])
                col = self.database[dict_index]

                
                self.surface.blit(self.pratino, (j* self.cellsize, i*self.cellsize))
                self.surface.blit(col, (j* self.cellsize, i*self.cellsize))
                
                if dict_index == "3.0":
                    self.surface.blit(self.peppone, (j* self.cellsize, i*self.cellsize))

        pygame.display.update()
        sleep(1)

    def __move(self, mossa):
        dati = mossa.split('_')
        
        if dati[0] == 'MOVE':
            x_attuale, y_attuale = dati[1].split(',')
            x_attuale = int(x_attuale)
            y_attuale = int(y_attuale)

            #print(f'Attuale: ({x_attuale}, {y_attuale})')

            x_successiva, y_successiva = dati[2].split(',')
            x_successiva = int(x_successiva)
            y_successiva = int(y_successiva)

            #print(f'Successiva: ({x_successiva}, {y_successiva})')

            dict_index_attuale = str(self.matrix[x_attuale][y_attuale])
            col_attuale = self.database[dict_index_attuale]

            dict_index_successivo = str(self.matrix[x_successiva][y_successiva])
            col_successivo = self.database[dict_index_successivo]

            rate = 40

            for i in range(rate):

                # ridisegna la cella attuale
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
                self.surface.blit(col_attuale, (y_attuale* self.cellsize, x_attuale*self.cellsize))

                # ridisegna la cella successiva
                self.surface.blit(self.pratino, (y_successiva* self.cellsize, x_successiva*self.cellsize))
                self.surface.blit(col_successivo, (y_successiva* self.cellsize, x_successiva*self.cellsize))

                # muove peppone
                x_move = x_successiva - x_attuale
                y_move = y_successiva - y_attuale

                if x_move == 0:
                    if y_move < 0:  # Sinistra
                        self.surface.blit(self.flip_walk_peppone, ((y_attuale*self.cellsize)-(self.cellsize/rate)*(i+1), x_attuale*self.cellsize))
                    elif y_move > 0:   # Destra
                        self.surface.blit(self.walk_peppone, ((y_attuale*self.cellsize)+(self.cellsize/rate)*(i+1), x_attuale*self.cellsize))
                elif y_move == 0:
                    if x_move < 0: # Su
                        self.surface.blit(self.walk_peppone, (y_attuale*self.cellsize, (x_attuale*self.cellsize)-(self.cellsize/rate)*(i+1)))
                    elif x_move > 0: # Giu
                        self.surface.blit(self.walk_peppone, (y_attuale*self.cellsize, (x_attuale*self.cellsize)+(self.cellsize/rate)*(i+1)))

                pygame.display.update()
                sleep(0.05)
            
        elif dati[0] == 'CLEAN':
            x_attuale, y_attuale = dati[1].split(',')
            x_attuale = int(x_attuale)
            y_attuale = int(y_attuale)

            cell_value = self.matrix[x_attuale][y_attuale]

            if cell_value == 1.:
                self.matrix[x_attuale][y_attuale] = 0.
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
            else:
                self.matrix[x_attuale][y_attuale] = 1.
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
                self.surface.blit(self.banana, (y_attuale* self.cellsize, x_attuale*self.cellsize))

            self.surface.blit(self.peppone, (y_attuale* self.cellsize, x_attuale*self.cellsize))
            
            pygame.display.update()
            sleep(2)
            
    def start(self):
        # prepara la finestra di pygame
        pygame.init()

        self.surface = pygame.display.set_mode((self.dimension * self.cellsize, self.dimension * self.cellsize))
        pygame.display.set_caption(self.title)

        self.__load_imgs()

        # stampa a video la matrice
        self.__print_GUI()

        # attende l'evento quit di pygame per terminare
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
            for solution in self.solutions:
                self.__move(solution)
            
            input()
            quit()
                



def main():
    # calcola la dimenzione delle celle
    # per adattare le varie matrici a schermo
    def findCellSize(n):
        return 300 - (n-3)*50

    
    matrix = [
        0., 0., 0., 0.,
        3., 5., 2., 0.,
        4., 1., 1., 0.,
        0., 0., 0., 0.
    ]

    matrix = [
        1., 1., 2.,
        5., 4., 5.,
        0., 0., 3.
    ]


    solutions = ['MOVE_2,2_2,1', 'MOVE_2,1_1,1', 'CLEAN_1,1', 'CLEAN_1,1', 'MOVE_1,1_0,1', 'CLEAN_0,1', 'MOVE_0,1_0,0', 'CLEAN_0,0', 'MOVE_0,0_0,1', 'MOVE_0,1_0,2']
    #solutions = ['MOVE_1,0_2,0', 'CLEAN_2,0', 'CLEAN_2,0', 'MOVE_2,0_2,1', 'CLEAN_2,1', 'MOVE_2,1_2,2', 'CLEAN_2,2', 'MOVE_2,2_2,1','MOVE_2,1_2,2', 'MOVE_2,2_2,1']

    n = int(sqrt(len(matrix)))

    matrix = np.array(matrix, dtype="float32").reshape(n, n)
    Animation(matrix, n, solutions, cellsize=findCellSize(n)).start()


if __name__ == "__main__":
    main()