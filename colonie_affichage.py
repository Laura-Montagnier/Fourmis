import numpy as np
import maze
import pheromone
import colony
import direction as d
import pygame as pg
from mpi4py import MPI
import sys
import time


globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank

#Les lignes qui suivent définissent des objets nécessaires à l'affichage.

#Initialise le module Pygame et la taille du labyrinthe
pg.init()
size_laby = 25, 25
#Au cas où on voudrait changer la taille du labyrinthe, on peut le mettre en argument.
if len(sys.argv) > 2:
    size_laby = int(sys.argv[1]),int(sys.argv[2])
#On envoie la taille au processus qui s'occupe de colonie_calculs.
globCom.send(size_laby, dest=1)


resolution = size_laby[1]*8, size_laby[0]*8
screen = pg.display.set_mode(resolution)
nb_ants = size_laby[0]*size_laby[1]//4
max_life = 500

if len(sys.argv) > 3:
    max_life = int(sys.argv[3])
pos_food = size_laby[0]-1, size_laby[1]-1
pos_nest = 0, 0

a_maze = maze.Maze(size_laby, 12345)
globCom.Bcast([a_maze.maze, MPI.INT8_T], root=0)


ants = colony.ColonyDisplay(nb_ants, pos_nest, max_life)
unloaded_ants = np.array(range(nb_ants))
alpha = 0.9
beta  = 0.99
if len(sys.argv) > 4:
    alpha = float(sys.argv[4])
if len(sys.argv) > 5:
    beta = float(sys.argv[5])


mazeImg = a_maze.display()
food_counter = 0

snapshop_taken = False
if food_counter == 1 and not snapshop_taken:
        pg.image.save(screen, "MyFirstFood.png")
        snapshop_taken = True

historic_path1 = ants.historic_path[0:51,:,:]
historic_path2 = ants.historic_path[52:103,:,:]
historic_path3 = ants.historic_path[104:155,:,:]

#La partie qui s'occupe de l'affichage est la suivante :

while True :   
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit(0)

    #On définit un "buffer" pherom avant d'y mettre quelque chose dedans  
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    globCom.Recv(pherom.pheromon, source=1)

    #Pour "importer" les fourmis, on a besoin de 3 tableaux différents.
    #On les importe chacun divisé en 3, ça fait 9 importations
    
    #all_historic_path = np.empty(nbp*(nbp-1),dtype=np.int64)
    #globCom.Allgather(colony.historic_path, all_historic_path)
    globCom.Recv([ants.historic_path, MPI.INT16_T], source=1)

    #globCom.Recv([historic_path1, MPI.INT16_T], source=1)
    #globCom.Recv([historic_path2, MPI.INT16_T], source=2)
    #globCom.Recv([historic_path3, MPI.INT16_T], source=3)

    

    globCom.Recv(ants.age, source=1)
    globCom.Recv(ants.directions, source=1)


    deb = time.time()
    #Utilise la surface screen pour y dessiner les phéromones :
    pherom.display(screen)
    #Permet de prendre une image et de la coller sur une autre :
    screen.blit(mazeImg, (0, 0))
    #Utilise la surface screen pour y dessiner des fourmis :
    ants.display(screen)
    #Fait l'affichage réel à l'écran :
    pg.display.update()

    end = time.time()

    print(f"Affichage en {end-deb} secondes.", end="\r")

