from asyncore import write
import itertools
import numpy as np
import random
import sys
import torch
from enviroment.moving_grid.utils.final_reward import write_firewall_file, generate_reward 
# Clase que genera el ambiente y permite interactuar con el
class FireGrid_V5:
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10, n_sims = 1, n_sims_final = 10):
        # Revisamos que la dimensión de la grilla sea divisible por 2
        assert size%2 == 0
        self.size = size # Dimensión de la grilla
        self.agent_dim = agent_dim # Dimensión del scope del agente
        self.agent_id = agent_id # Número que representa las celdas en donde está el agente
        self._space = torch.zeros(3, self.size, self.size) # Privado, corresponde al espacio del ambiente -> Tensor de 3 x size x size
        self._agent_location = np.zeros(2, dtype=int) # Privado, ubicación de la primera entrada del scope del agente 
        self._random = random # generador de numeros aleatorios
        self._action_map = {} # Diccionario que mapea el identificador de una acción con la combinación asociada, ej: 0 -> (0, 0, 0, 0)
        combinations = list(itertools.product([0, -1], repeat=agent_dim**2))
        for i in range(2**(agent_dim**2)):
            self._action_map[i] = combinations[i]
        self.burn_value = burn_value
        self.n_sims = n_sims
        self.n_sims_final = n_sims_final
        # Se incorpora la información correspondiente al tipo de combustible
        forest_data = np.loadtxt("/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/data/Sub20x20/Forest.asc", skiprows=6)
        norm = np.linalg.norm(forest_data)
        forest_data = forest_data/norm
        self._space[2] = torch.Tensor(forest_data)

    def get_space(self):
        """Función que retorna el espacio del ambiente"""
        return self._space

    def get_agent_location(self):
        """Función que retorna la ubicación del agente"""
        return self._agent_location
    
    def get_action_map(self):
        """Función que retorna el mapa de acciones"""
        return self._action_map

    def get_space_dims(self):
        """Función que retorna las dimensiones del espacio"""
        return (self.size, self.size)

    def get_action_space_dims(self):
        """Función que retorna las dimensiones del espacio de acciones"""
        return (len(self._action_map), 1)

    def mark_agent(self):
        """Función que dada la ubicación del agente, marca esta en la grilla"""
        self._space[1][self._agent_location[0], self._agent_location[1]] = 1
        self._space[1][self._agent_location[0], self._agent_location[1] + 1] = 1
        self._space[1][self._agent_location[0] +1, self._agent_location[1]] = 1
        self._space[1][self._agent_location[0] + 1, self._agent_location[1] + 1] = 1
    def erase_agent(self):
        """Función que dada la ubicación del agente, borra el agente de la grilla"""
        self._space[1][self._agent_location[0], self._agent_location[1]] = 0
        self._space[1][self._agent_location[0], self._agent_location[1] + 1] = 0
        self._space[1][self._agent_location[0] +1, self._agent_location[1]] = 0
        self._space[1][self._agent_location[0] + 1, self._agent_location[1] + 1] = 0
    def random_action(self):
        """Función que retorna una acción aleatoria del espacio de acciones"""
        return torch.Tensor([self._random.randint(0, 15)])

    def set_seed(self, seed):
        """Función que establece una semilla para los numeros aleatorios"""
        self._random.seed(seed)

    def reset(self):
        """Función que resetea el ambiente -> Agente en posición (0,0) y toda la grilla sin marcar"""
        self._space[0] = torch.zeros(self.size, self.size)
        self._space[1] = torch.zeros(self.size, self.size)
        self._agent_location[0] = 0
        self._agent_location[1] = 0
        self.mark_agent()
        return self._space

    def step(self, action):
        """Función que genera la transicion del ambiente desde un estado a otro generada por la acción action. Retorna: espacio, reward, done"""
        action_tuple = self._action_map[action.item()] # Se pasa la acción a la combinación de celdas correspondiente
        # Calculamos la penalización por marcar celdas
        r = 0
        done = False
        for i in action_tuple:
            if i == -1:
                r -= 1
        # Marcamos las celdas seleccionadas en la matriz de cortafuegos
        self._space[0][self._agent_location[0], self._agent_location[1]] = action_tuple[0]
        self._space[0][self._agent_location[0], self._agent_location[1] + 1] = action_tuple[1]
        self._space[0][self._agent_location[0] +1, self._agent_location[1]] = action_tuple[2]
        self._space[0][self._agent_location[0] + 1, self._agent_location[1] + 1] = action_tuple[3]
        # Borramos el agente
        self.erase_agent()
        # Actualizamos la posición del agente
        if self._agent_location[0] < self.size - self.agent_dim:
            self._agent_location[0] += self.agent_dim
        elif self._agent_location[1] < self.size - self.agent_dim:
            self._agent_location[0] = 0
            self._agent_location[1] += self.agent_dim
        else:
            # Llegamos al final de la grilla, marcamos done = True y generamos la recompensa
            done = True
            # Escribimos el archivo que se pasa al simulador 
            write_firewall_file(self._space[0])
            # Se lo pasamos al simulador y obtenemos la recompensa
            reward = generate_reward(self.n_sims_final)*self.burn_value + r
            s = self._space
            return s, torch.Tensor([reward]).reshape((1, 1)), done
        # Escribimos el archivo que se pasa al simulador 
        write_firewall_file(self._space[0])
        # Se lo pasamos al simulador y obtenemos la recompensa
        reward = generate_reward(self.n_sims)*self.burn_value + r
        self.mark_agent()
        s = self._space
        return s, torch.Tensor([reward]).reshape((1, 1)), done
    
    def sample_space(self):
        """Función que genera un estado aleatorio del sistema"""
        self.reset()
        iterations = self._random.randint(0, (self.size/self.agent_dim)**2)
        for i in range(iterations):
            self.step(self.random_action())
        return self._space
    def show_state(self):
        """Función que printea el estado del ambiente"""
        print(f"Posicion del agente: {self._agent_location}")
        print(f"Estado de la grilla del bosque:")
        print(self._space[0])
        print(f"Estado de la grilla del agente:")
        print(self._space[1])
        print(f"Estado de la grilla del Bosque:")
        print(self._space[2])
        print("\n")

