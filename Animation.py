import numpy as np
import pygame


class Animation:
    def __init__(self, matrix: np.array, dimension: int, cellsize=50, title="Typing Monkeys's Deepest Snake"):
        # la matrice viene copiata per evitare che venga modificata anche al di fuore di questa classe
        self.matrix = np.copy(matrix)

        # dimensioni della stampa a video
        self.dimension = dimension
        self.cellsize = cellsize

        # titolo della finestra
        self.title = title

        self.col_grid = (255, 255, 255)
        self.surface = None

        # contiene le relazioni Numero-Lettera e Lettera-Numer
        # Nelle posizioni del dizionario rappresentate da numeri ('2.0') ci sono il nome della corrispettiva 
        # lettera e il colore che la rappresenta.
        # 
        # Nelle posizione rappresentate dalle lettere ('c') ci sono i corrispettivi numeri utilizzati
        # per codificare le lettere. Queste ultime posizioni sono solo per creare la matrice a mano, in quanto
        # nel problema reale avremmo una matrice gia' popolata da numeri e non andra' creata.
        self.database = {
            '0.0': {
                'name': '-',
                'col': (0, 0, 0)
            },

            '2.0': {
                'name': 'c',
                'col': (255, 255, 255)
            },

            '3.0': {
                'name': 'd',
                'col': (210,180,140)
            },

            '5.0': {
                'name': 'f',
                'col': (0, 0, 255)

            },

            '18.0': {
                'name': 's',
                'col': (0, 255, 0)
            },

            '21.0': {
                'name': 'v',
                'col': (139, 69, 19)
            },

            '23.0': {
                'name': 'x',
                'col': (0, 0, 0)
            },

            'c': 2.0,
            'd': 3.0,
            'f': 5.0,
            's': 18.0,
            'v': 21.0,
            'x': 23.0
        }

    def __print_GUI(self):
        self.surface.fill(self.col_grid)

        for i in range(self.dimension):
            for j in range(self.dimension):
                # colora la cella in base al suo contenuto
                col = self.database[str(self.matrix[i][j])]['col']

                # stampa la cella corrente
                pygame.draw.rect(self.surface, col, (j*self.cellsize, i*self.cellsize, self.cellsize-1, self.cellsize-1))

        pygame.display.update()

    def start(self):
        # prepara la finestra di pygame
        pygame.init()
        self.surface = pygame.display.set_mode((self.dimension * self.cellsize, self.dimension * self.cellsize))
        pygame.display.set_caption(self.title)

        # stampa a video la matrice
        self.__print_GUI()

        # attende l'evento quit di pygame per terminare
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

