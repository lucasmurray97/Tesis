import numpy as np
import torch
from enviroment.abstract_moving_grid import Moving_Grid
# Clase que genera el ambiente y permite interactuar con el
class FireGrid_V8(Moving_Grid):
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10, n_sims = 1, n_sims_final = 10, env_id = 0):
        super().__init__(size, burn_value, agent_id, agent_dim, n_sims_final, env_id = 0)
        self.n_sims = n_sims
        self.last_complete = torch.zeros(20,20)

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

        # Actualizamos last_complete
        self.last_complete[self._agent_location[0], self._agent_location[1]] = action_tuple[0]
        self.last_complete[self._agent_location[0], self._agent_location[1] + 1] = action_tuple[1]
        self.last_complete[self._agent_location[0] +1, self._agent_location[1]] = action_tuple[2]
        self.last_complete[self._agent_location[0] + 1, self._agent_location[1] + 1] = action_tuple[3]

        # Marcamos en la grilla de combustibles 1 si en la posición se marcó un cortafuego
        if action_tuple[0] == -1:
            self._space[2][self._agent_location[0], self._agent_location[1]] = 1
        if action_tuple[1] == -1:
            self._space[2][self._agent_location[0], self._agent_location[1] + 1] = 1
        if action_tuple[2] == -1:            
            self._space[2][self._agent_location[0] +1, self._agent_location[1]] = 1
        if action_tuple[3] == -1:
            self._space[2][self._agent_location[0] + 1, self._agent_location[1] + 1] = 1
        # Borramos al agente
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
            if self.size < 20:
                firegrid = self.up_scale(self.last_complete, 20)
            else:
                firegrid = self.last_complete
            self.write_firewall(firegrid)
            # Se lo pasamos al simulador y obtenemos la recompensa
            reward = self.generate_reward(self.n_sims_final) + r
            # Retornamos la matriz de posición y de combustibles
            s = self._space[1:]
            return s, torch.Tensor([reward]).reshape((1, 1)), done
        # Escribimos el archivo que se pasa al simulador, esta véz con last_complete
        if self.size < 20:
                firegrid = self.up_scale(self.last_complete, 20)
        else:
            firegrid = self.last_complete
        self.write_firewall(firegrid)
        # Se lo pasamos al simulador y obtenemos la recompensa
        reward = self.generate_reward(self.n_sims) + r
        # Marcamos al agente
        self.mark_agent()
        # Retornamos la matriz de posición y de combustibles
        s = self._space[1:]
        return s, torch.Tensor([reward]).reshape((1, 1)), done
    


