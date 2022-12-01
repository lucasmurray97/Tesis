import numpy as np
import torch
from moving_grid.abstract_moving_gym import Abstract_Moving_Gym

class Moving_Gym_v6(Abstract_Moving_Gym):
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10, n_sims_final = 10):
        super().__init__(size, burn_value, agent_id, agent_dim, n_sims_final)
    
    def step(self, action):
        """Función que genera la transicion del ambiente desde un estado a otro generada por la acción action. Retorna: espacio, reward, done"""
        action_tuple = self._action_map[action.item()] # Se pasa la acción a la combinación de celdas correspondiente
        # Calculamos la penalización por marcar celdas
        r = 0 
        for i in action_tuple:
            if i == -1:
                r -= 1
        # Marcamos las celdas seleccionadas en la matriz de cortafuegos
        self.firegrid[self._agent_location[0], self._agent_location[1]] = action_tuple[0]
        self.firegrid[self._agent_location[0], self._agent_location[1] + 1] = action_tuple[1]
        self.firegrid[self._agent_location[0] +1, self._agent_location[1]] = action_tuple[2]
        self.firegrid[self._agent_location[0] + 1, self._agent_location[1] + 1] = action_tuple[3]
        # Marcamos en la grilla de combustibles 1 si en la posición se marcó un cortafuego
        if action_tuple[0] == -1:
            self.observation_space[1][self._agent_location[0], self._agent_location[1]] = 1
        if action_tuple[1] == -1:
            self.observation_space[1][self._agent_location[0], self._agent_location[1] + 1] = 1
        if action_tuple[2] == -1:            
            self.observation_space[1][self._agent_location[0] +1, self._agent_location[1]] = 1
        if action_tuple[3] == -1:
            self.observation_space[1][self._agent_location[0] + 1, self._agent_location[1] + 1] = 1
        # Borramos el agente
        self.erase_agent()
        # Marcamos la posición del agente en la matriz de posición
        if self._agent_location[0] < self.size - self.agent_dim:
            self._agent_location[0] += self.agent_dim
        elif self._agent_location[1] < self.size - self.agent_dim:
            self._agent_location[0] = 0
            self._agent_location[1] += self.agent_dim
        else:
            # Llegamos al final de la grilla, marcamos done = True y generamos la recompensa
            # Escribimos el archivo que se pasa al simulador 
            if self.size < 20:
                self.firegrid = self.up_scale(self.firegrid, 20)
            self.write_firewall(self.firegrid)
            # Se lo pasamos al simulador y obtenemos la recompensa
            final_reward = self.generate_reward(self.n_sims_final) + r
            # Retornamos la matriz de posición y de combustibles
            return self.observation_space, torch.Tensor([final_reward]).reshape((1, 1)),  True
        # Marcamos el agente
        self.mark_agent()
        # Retornamos la matriz de posición y de combustibles
        return self.observation_space, torch.Tensor([r]).reshape((1, 1)),  False