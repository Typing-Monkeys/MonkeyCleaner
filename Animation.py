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
        self.cellsize = self.__findCellSize()

        # array con tutte le mosse per la soluzione finale
        self.solutions = solutions

        # titolo della finestra
        self.title = title

        self.col_grid = (255, 255, 255)
        self.surface = None

        # indice per avanzare progressivamente le varie mosse
        self.index = 0

    # calcola la dimenzione appropriata di cellsize
    # in base alla dimenzione della matrice
    def __findCellSize(self):
        return 300 - (self.dimension-3)*50

    # funzione di classe per fare il load delle varie immagini per l'animazione
    def __load_imgs(self):
        # l'immagine viene caricata con le sue dimenzioni originiali e poi
        # resa delle dimenzioni giuste per essere disegnata nelle celle

        # sfondo start (soggiorno)
        self.startbg = pygame.image.load('./Animation_imgs/Appartamento_Scandinavo_di_Peppone.png').convert_alpha()
        self.startbg = pygame.transform.scale(self.startbg, (self.cellsize, self.cellsize))

        # sfondo stop (camera da letto)
        self.stopbg = pygame.image.load('./Animation_imgs/Letto_Scandinavo_di_Peppone.png').convert_alpha()
        self.stopbg = pygame.transform.scale(self.stopbg, (self.cellsize, self.cellsize))

        # singola banana (dirty)
        self.banana = pygame.image.load('./Animation_imgs/Banana.png').convert_alpha()
        self.banana = pygame.transform.scale(self.banana, (self.cellsize, self.cellsize))
        
        # doppia banana (very_derty)
        self.bananas = pygame.image.load('./Animation_imgs/Banana_Doppia.png').convert_alpha()
        self.bananas = pygame.transform.scale(self.bananas, (self.cellsize, self.cellsize))

        # muro (casella non accessibile)
        self.wall = pygame.image.load('./Animation_imgs/Foresta.png').convert_alpha()
        self.wall = pygame.transform.scale(self.wall, (self.cellsize, self.cellsize))
        
        # stanza pulita (clean)
        self.pratino = pygame.image.load('./Animation_imgs/Pratino.png').convert_alpha()
        self.pratino = pygame.transform.scale(self.pratino, (self.cellsize, self.cellsize))
        
        # frame di inizio, fine e stato mangia-banana di Peppone
        self.peppone = pygame.image.load('./Animation_imgs/Peppone.png').convert_alpha()
        self.peppone = pygame.transform.scale(self.peppone, (self.cellsize, self.cellsize))

        # frame di Peppone mentre si sposta tra stanze (verso destra)
        self.walk_peppone = pygame.image.load('./Animation_imgs/Walking_Peppone.png').convert_alpha()
        self.walk_peppone = pygame.transform.scale(self.walk_peppone, (self.cellsize, self.cellsize))
        
        # frame di Peppone per andare a sinistra
        # (fa un flip del'immagine)
        self.flip_walk_peppone = pygame.transform.flip(self.walk_peppone, True, False)

        # frame della scimmia che pensa ad una soluzione (il tempo impiegato nell'animazione non è quello del vero calcolo)
        self.neuron_activation1 = pygame.image.load('./Animation_imgs/Neuron_Activation1.png').convert_alpha()
        self.neuron_activation1 = pygame.transform.scale(self.neuron_activation1, (self.cellsize, self.cellsize))

        # frame della scimmia che ha finito di trovare una soluzione e si accinge ad eseguirla
        self.neuron_activation2 = pygame.image.load('./Animation_imgs/Neuron_Activation2.png').convert_alpha()
        self.neuron_activation2 = pygame.transform.scale(self.neuron_activation2, (self.cellsize, self.cellsize))


        # contiene le relazioni Numero-Lettera e Lettera-Numer
        # Nelle posizioni del dizionario rappresentate da numeri ('2.0') c'è 
        # l'immagine che andrà stampata a video
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

    # funzione per dare un delay alle animazioni
    # nel mentre controlla gli eventi nella finestra 
    # di pygame per evitare freeze del programma
    def __aspetta(self, cicli: int, tempo: int):
        for _ in range(cicli):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            pygame.time.wait(tempo)

    # stampa la fase iniziale dell'animazione
    def __print_GUI(self):
        # caricamento e riproduzione della "soundtrack" mentre si svolge la soluzione
        pygame.mixer.music.load('./Animation_imgs/Monkey_beat.mp3')
        pygame.mixer.music.play(-1, 8.)
        pygame.mixer.music.set_volume(0.1)

        # salva la posizione iniziale di Peppone
        # per usarla quando verrà fatta l'animazione del NeuronActivation
        peppone_index = None

        # disegna lo stato iniziale del problema
        for i in range(self.dimension):
            for j in range(self.dimension):
                # colora la cella in base al suo contenuto
                dict_index = str(self.matrix[i][j])
                col = self.database[dict_index]

                # disegna sempre lo sfondo e quello e il vero contenuto della cella
                self.surface.blit(self.pratino, (j* self.cellsize, i*self.cellsize))
                self.surface.blit(col, (j* self.cellsize, i*self.cellsize))
                
                # se la casella è la casella di inizio viene disegnato anche peppone su di essa
                if dict_index == "3.0":
                    peppone_index = (i, j)
                    self.surface.blit(self.peppone, (j* self.cellsize, i*self.cellsize))

        # attesa di 5 secondi per poter fare testing e streaming senza perdersi nessun frame importante
        pygame.display.update()
        self.__aspetta(1000, 5)
        
        # prima fase dell'animazione NeuronActivation. Qui Peppone pensa
        self.surface.blit(self.startbg, (peppone_index[1]* self.cellsize, peppone_index[0]*self.cellsize))
        self.surface.blit(self.neuron_activation1, (peppone_index[1]* self.cellsize, peppone_index[0]*self.cellsize))
        pygame.display.update()
        self.__aspetta(1000, 3)

        # seconda fase dell'animazione NeuronActivation. Qui si attiva il neurone della scimmia !
        self.surface.blit(self.startbg, (peppone_index[1]* self.cellsize, peppone_index[0]*self.cellsize))
        self.surface.blit(self.neuron_activation2, (peppone_index[1]* self.cellsize, peppone_index[0]*self.cellsize))
        pygame.display.update()
        self.__aspetta(900, 2)
    
    # in base alla mossa passata deduce l'azione da fare e la stampa a video
    def __move(self, mossa: str):
        # prende i valori della mossa
        dati = mossa.split('_')
        
        # controlla che sia un azione di movimento
        if dati[0] == 'MOVE':
            # prende le coordinate attuali
            x_attuale, y_attuale = dati[1].split(',')
            x_attuale = int(x_attuale)
            y_attuale = int(y_attuale)

            # prende le coordinate delle cella in cui deve andare
            x_successiva, y_successiva = dati[2].split(',')
            x_successiva = int(x_successiva)
            y_successiva = int(y_successiva)

            # Ogni volta vanno aggiornate la cella attuale e quella successiva per
            # evitare di lasciare qualche frame dell'animazione di movimento
            # sempre visibile su una delle due celle coinvolte nello spostamento

            # prende l'immagine della cella attuale
            dict_index_attuale = str(self.matrix[x_attuale][y_attuale])
            col_attuale = self.database[dict_index_attuale]
            
            # prende l'immagine della cella successiva
            dict_index_successivo = str(self.matrix[x_successiva][y_successiva])
            col_successivo = self.database[dict_index_successivo]
            
            # frequenza con cui verranno aggionrati i frame dell'animazione
            rate = 40

            # calcolo per dedurre in seguito lo spostamento da fare e l'animazione corrispondente
            x_move = x_successiva - x_attuale
            y_move = y_successiva - y_attuale

            # disegna l'animazione della camminata
            for i in range(rate):
                # ridisegna la cella attuale
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
                self.surface.blit(col_attuale, (y_attuale* self.cellsize, x_attuale*self.cellsize))

                # ridisegna la cella successiva
                self.surface.blit(self.pratino, (y_successiva* self.cellsize, x_successiva*self.cellsize))
                self.surface.blit(col_successivo, (y_successiva* self.cellsize, x_successiva*self.cellsize))

                # utilizza x_move e y_move per dedurre dove muoversi e ne stampa un frame
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
                
                # attesa per rendere l'animazione più "fluida"
                pygame.display.update()
                self.__aspetta(20, 1)
        
        # controlla che sia un'azione di pulizia
        elif dati[0] == 'CLEAN':
            # prende le coordinate della cella attuale
            x_attuale, y_attuale = dati[1].split(',')
            x_attuale = int(x_attuale)
            y_attuale = int(y_attuale)

            # vede se la cella è DIRTY o VERYDIRTY
            cell_value = self.matrix[x_attuale][y_attuale]

            # se DIRTY
            if cell_value == 1.:
                # pulisce la cella e disegna l'animazione di pulizia
                self.matrix[x_attuale][y_attuale] = 0.
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
            # se VERYDIRTY
            else:
                # rende la cella DIRTY e disegna l'animazione di pulizia una sola volta
                self.matrix[x_attuale][y_attuale] = 1.
                self.surface.blit(self.pratino, (y_attuale* self.cellsize, x_attuale*self.cellsize))
                self.surface.blit(self.banana, (y_attuale* self.cellsize, x_attuale*self.cellsize))

            # disegna Peppone che mangia
            self.surface.blit(self.peppone, (y_attuale* self.cellsize, x_attuale*self.cellsize))
            
            # attesa di Peppone che mangia
            pygame.display.update()
            self.__aspetta(750, 2)

    # effettua l'azione successiva
    def __nextmove(self):
        # controlla che ci siano ancora azioni da fare
        # se si sono concluse allora stampa Peppone (il frame finale di Peppone seduto sul letto)
        if self.index >= len(self.solutions):
            # interrompe il While principale
            self.alive = False
            
            # salva le coordinate della cella dell'ultima mossa
            # (ogni volta l'ultima mossa sarà spostarsi nella cella finale)
            x, y = self.solutions[-1].split('_')[-1].split(',')
            x = int(x)
            y = int(y)
            
            # stampa Peppone nella cella finale
            self.surface.blit(self.stopbg, (y* self.cellsize, x*self.cellsize))
            self.surface.blit(self.peppone, (y* self.cellsize, x*self.cellsize))
            pygame.display.update()
            
            # attesa per ammirare Peppone dopo aver concluso il suo arduo lavoro!
            self.__aspetta(1000, 2)

            # chiude la finestra di pygame
            pygame.quit()
            return
        
        # prende la prossima azione
        azione = self.solutions[self.index]
        
        # esegue l'azione
        self.__move(azione)

        # l'indice punta all'azione successiva
        self.index += 1

    # avvia l'animazione
    def start(self):
        # prepara la finestra di pygame
        pygame.init()

        self.surface = pygame.display.set_mode((self.dimension * self.cellsize, self.dimension * self.cellsize))
        pygame.display.set_caption(self.title)

        self.__load_imgs()

        # stampa a video la matrice
        self.__print_GUI()
        
        self.alive = True

        # attende l'evento quit di pygame per terminare
        while self.alive: 
            # controlla gli eventi di pygame
            # (va fatto sennò si rompe)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.__nextmove()
