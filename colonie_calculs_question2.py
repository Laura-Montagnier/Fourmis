import numpy as np
import pheromone
import colony
import direction as d
from mpi4py import MPI

UNLOADED, LOADED = False, True

exploration_coefs = 0.

globCom = MPI.COMM_WORLD.Dup()
nbp     = globCom.size
rank    = globCom.rank



import sys
import time

#On récupère la taille du labyrinthe qui est définie par l'affichage.
size_laby = None
size_laby = globCom.recv(source=0)

nb_ants = size_laby[0]*size_laby[1]//4 #On définit ici le nombre de fourmis.
max_life = 500

#On assigne un nombre de fourmis par processeur
#On se simplifie la vie en estimant que le compte est rond.
assert nb_ants%(nbp-1) == 0;
nb_local_fourmis = nb_ants//(nbp-1) #Il y aura 52 fourmis par mini-colonie.


#Au cas où l'utilisateur veuille changer la durée de vie max.
if len(sys.argv) > 3:
    max_life = int(sys.argv[3])


#Le nid est en haut à gauche, la nourriture est en bas à droite.
pos_food = size_laby[0]-1, size_laby[1]-1
pos_nest = 0, 0

#On initialise le labyrinthe.
a_maze = np.zeros(size_laby, dtype=np.int8)
globCom.Bcast([a_maze, MPI.INT8_T], root=0)

#On initialise les fourmis.
ants = colony.ColonyCalcul(nb_local_fourmis, pos_nest, max_life)
unloaded_ants = np.array(range(nb_ants))


#Au cas où l'utilisateur veuille changer les valeurs alpha et beta.
alpha = 0.9
beta  = 0.99
if len(sys.argv) > 4:
    alpha = float(sys.argv[4])
if len(sys.argv) > 5:
    beta = float(sys.argv[5])


#On initialise les phéromones et on initialise le compteur de nourriture à zéro.
pherom = pheromone.Pheromon(size_laby, pos_food, alpha, beta)
food_counter = 0

#Nombre de données transférées :
nb_data=np.array((0,52,52,52))
nb_data_1=501*2*nb_data


snapshop_taken = False
while True:
    #On doit mettre ces lignes dans le "while" pour envoyer à chaque 
    #fois qu'on en a besoin les informations nécessaires à l'affichage.


    globCom.Gatherv(ants.directions, (ants.directions, nb_data), root=0)

    globCom.Gatherv(ants.age, (ants.age, nb_data), root=0)

    globCom.Gatherv(ants.historic_path, (ants.historic_path, nb_data_1), root=0)

    

    #globCom.Send([ants.historic_path, MPI.INT16_T], dest=0)
    #globCom.Send([ants.age, MPI.INT16_T], dest=0)
    #globCom.Send([ants.directions, MPI.INT8_T], dest=0)

    
    globCom.Send([pherom.pheromon, MPI.INT16_T], dest=0) #il faut partager les phéromones


    deb = time.time()
    food_counter = ants.advance(a_maze, pos_food, pos_nest, pherom, food_counter)
    pherom.do_evaporation(pos_food)
    end = time.time()


    print(f"FPS : {1./(end-deb):6.2f}, nourriture : {food_counter:7d}", end='\r')

