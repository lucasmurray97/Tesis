from asyncore import write
import itertools
import numpy as np
import random
import sys
from enviroment.utils.final_reward import write_firewall_file, generate_reward 
# Clase que genera el ambiente y permite interactuar con el
class FireGrid_V2:
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10):
        # Revisamos que la dimensión de la grilla sea divisible por 2
        assert size%2 == 0
        self.size = size # Dimensión de la grilla
        self.agent_dim = agent_dim # Dimensión del scope del agente
        self.agent_id = agent_id # Número que representa las celdas en donde está el agente
        self._space = np.zeros((self.size, self.size), dtype=int) # Privado, corresponde al espacio del ambiente -> Matriz de size x size
        self._agent_location = np.zeros(2, dtype=int) # Privado, ubicación de la primera entrada del scope del agente 
        self._random = random # generador de numeros aleatorios
        self._action_map = {} # Diccionario que mapea el identificador de una acción con la combinación asociada, ej: 0 -> (0, 0, 0, 0)
        combinations = list(itertools.product([0, -1], repeat=agent_dim**2))
        for i in range(2**(agent_dim**2)):
            self._action_map[i] = combinations[i]
        self.burn_value = burn_value
        self.initial_pos = self._random.randint(0, 1) # if 0 -> (0,0) else -> (20, 20)
        if self.initial_pos == 1:
            self._agent_location = np.array([self.size - 2, self.size - 2,], dtype=int)
        self.pace_direction = self._random.randint(0, 1)
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
        return (self.size, self.size)

    def get_action_space_dims(self):
        return (len(self._action_map), 1)
    
    def get_initial_pos(self):
        return self.initial_pos

    def get_pace_direction(self):
        return self.pace_direction

    def mark_agent(self):
        """Función que dada la ubicación del agente, marca esta en la grilla"""
        self._space[self._agent_location[0], self._agent_location[1]] = 1
        self._space[self._agent_location[0], self._agent_location[1] + 1] = 1
        self._space[self._agent_location[0] + 1, self._agent_location[1]] = 1
        self._space[self._agent_location[0] + 1, self._agent_location[1] + 1] = 1
    def random_action(self):
        """Función que retorna una acción aleatoria del espacio de acciones"""
        return self._random.randint(0, 15)

    def set_seed(self, seed):
        """Función que establece una semilla para los numeros aleatorios"""
        self._random.seed(seed)

    def reset(self):
        """Función que resetea el ambiente -> Agente en posición (0,0) y toda la grilla sin marcar"""
        self._space = np.zeros((self.size, self.size), dtype=int)
        if self.initial_pos == 0:
            self._agent_location[0] = 0
            self._agent_location[1] = 0
        else: 
            self._agent_location[0] = 18
            self._agent_location[1] = 18
        self.mark_agent()
        return self._space
    
    def mark_action(self, action):
        action_tuple = self._action_map[action]
        self._space[self._agent_location[0], self._agent_location[1]] = action_tuple[0]
        self._space[self._agent_location[0], self._agent_location[1] + 1] = action_tuple[1]
        self._space[self._agent_location[0] +1, self._agent_location[1]] = action_tuple[2]
        self._space[self._agent_location[0] + 1, self._agent_location[1] + 1] = action_tuple[3]

    def move_agent_vertical(self, action):
        action_tuple = self._action_map[action]
        r = 0
        for i in action_tuple:
            if i == -1:
                r -= 1
        if self.initial_pos == 0:
            if self._agent_location[0] < self.size - self.agent_dim:
                self._agent_location[0] += self.agent_dim
            elif self._agent_location[1] < self.size - self.agent_dim:
                self._agent_location[0] = 0
                self._agent_location[1] += self.agent_dim
            else:
                write_firewall_file(self._space)
                final_reward = generate_reward()*self.burn_value
                return self._space, final_reward,  True  
        else:
            if self._agent_location[0] > 0:
                self._agent_location[0] -= self.agent_dim
            elif self._agent_location[1] > 0:
                self._agent_location[0] = self.size - self.agent_dim
                self._agent_location[1] -= self.agent_dim
            else:
                write_firewall_file(self._space)
                final_reward = generate_reward()*self.burn_value
                return self._space, final_reward,  True  
        return self._space, r, False

    def move_agent_horizontal(self, action):
        action_tuple = self._action_map[action]
        r = 0
        for i in action_tuple:
            if i == -1:
                r -= 1
        if self.initial_pos == 0:
            if self._agent_location[1] < self.size - self.agent_dim:
                self._agent_location[1] += self.agent_dim
            elif self._agent_location[0] < self.size - self.agent_dim:
                self._agent_location[1] = 0
                self._agent_location[0] += self.agent_dim
            else:
                write_firewall_file(self._space)
                final_reward = generate_reward()*self.burn_value
                return self._space, final_reward,  True  
        else:
            if self._agent_location[1] > 0:
                self._agent_location[1] -= self.agent_dim
            elif self._agent_location[0] > 0:
                self._agent_location[1] = self.size - self.agent_dim
                self._agent_location[0] -= self.agent_dim
            else:
                write_firewall_file(self._space)
                final_reward = generate_reward()*self.burn_value
                return self._space, final_reward,  True  
        return self._space, r, False

    def move_agent(self, action):
        if self.pace_direction == 0:
            return self.move_agent_vertical(action)
        else:
            return self.move_agent_horizontal(action)


    def step(self, action):
        """Función que genera la transicion del ambiente desde un estado a otro generada por la acción action. Retorna: espacio, reward, done"""
        self.mark_action(action)
        space, reward, done = self.move_agent(action)
        if not done:
            self.mark_agent()
        return space, reward, done
    
    def sample_space(self):
        self.reset()
        iterations = self._random.randint(0, (self.size/self.agent_dim)**2)
        for i in range(iterations):
            self.step(self.random_action())
        return self._space
    def show_state(self):
        """Función que printea el estado del ambiente"""
        print(f"Posicion del agente: {self._agent_location}")
        print(f"Estado de la grilla:")
        print(self._space)
        print("\n")

