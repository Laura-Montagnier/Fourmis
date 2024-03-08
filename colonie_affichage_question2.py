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

#On assigne un nombre de fourmis par processeur
#On se simplifie la vie en estimant que le compte est rond.
assert nb_ants%(nbp-1) == 0;
nb_local_fourmis = nb_ants//(nbp-1) #Il y aura 52 fourmis par mini-colonie.

#Nombre de données transférées :
nb_data=np.array((0,52,52,52))
nb_data_1=501*2*nb_data

#La partie qui s'occupe de l'affichage est la suivante :

while True :   
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit(0)

    #On définit un "buffer" pherom avant d'y mettre quelque chose dedans  
    pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
    
    #A quelles tailles est-ce que l'on s'attend ? Ici, on remplace nb_ants par le nombre local.
    #historic_path = np.zeros((nb_ants, max_life+1, 2), dtype=np.int16)
    #age = np.zeros(nb_ants, dtype=np.int64)
    #directions = d.DIR_NONE*np.ones(nb_ants, dtype=np.int8)
    #pheromon = np.zeros((the_dimensions[0]+2, the_dimensions[1]+2), dtype=np.double)

    
    
    print("Saluuut")

    globCom.Gatherv(ants.directions, (ants.directions, nb_data), root=0)

    
    globCom.Gatherv(ants.age, (ants.age, nb_data), root=0)
    
    globCom.Gatherv(ants.historic_path, (ants.historic_path, nb_data_1), root=0)
    
    globCom.Recv(pherom.pheromon, source=1)

    #globCom.Recv([ants.historic_path, MPI.INT16_T], source=rank+1)
    #globCom.Recv(ants.age, source=rank+1)
    #globCom.Recv(ants.directions, source=rank+1)
    


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

