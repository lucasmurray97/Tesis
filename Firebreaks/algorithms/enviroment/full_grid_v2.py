import torch
import torch.nn.functional as F
from enviroment.abstract_full_grid import Abstract_Full_Grid
import torch.nn.functional as F
import numpy as np
class Full_Grid_V2(Abstract_Full_Grid):
    def __init__(self, size, burn_value = 10, n_sims_final = 50, env_id = 0, instance = "homo_1"):
        super().__init__(size, burn_value, n_sims_final, env_id, instance)
        self.actions_history = torch.ones(self.size**2, dtype=torch.bool)
        self.shape = (2, self.size, self.size)
        self._space = torch.zeros(3, self.size, self.size)
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        self._step = 0
        self.marked = 0
        self.init_conn_layer = self.create_conectivity_layer()
        self._space[2] = self.init_conn_layer
        

    def create_conectivity_layer(self):
        connectivity_layer = torch.zeros(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                if (i == 0 and j == 0) or (i == 0 and j == self.size -1) or (i == self.size -1 and j == 0) or (i == self.size-1 and j == self.size - 1):
                    connectivity_layer[i,j] = 3/8
                elif i == 0 or i == self.size-1 or j == 0 or j==self.size-1:
                    connectivity_layer[i,j] = 5/8
                else:
                    connectivity_layer[i,j] = 1.
        return connectivity_layer

    def mark_connectivity_layer(self, position):
        i, j = position
        self._space[2][i,j] = 0.
        if i == 0 and j == 0:
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]
            self._space[2][i+1,j+1] = self._space[2][i+1,j+1] - 1/8 if self._space[2][i+1,j+1] > 0 else self._space[2][i+1,j+1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
        elif i == 0 and j == self.size-1:
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i+1,j-1] = self._space[2][i+1,j-1] - 1/8 if self._space[2][i+1,j-1] > 0 else self._space[2][i+1,j-1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
        elif i == self.size-1 and j == 0:
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]
            self._space[2][i-1,j+1] = self._space[2][i-1,j+1] - 1/8 if self._space[2][i-1,j+1] > 0 else self._space[2][i-1,j+1]
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
        elif i == self.size-1 and j == self.size-1:
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i-1,j-1] = self._space[2][i-1,j-1] - 1/8 if self._space[2][i-1,j-1] > 0 else self._space[2][i-1,j-1]
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
        elif i == 0:
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i+1,j-1] = self._space[2][i+1,j-1] - 1/8 if self._space[2][i+1,j-1] > 0 else self._space[2][i+1,j-1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
            self._space[2][i+1,j+1] = self._space[2][i+1,j+1] - 1/8 if self._space[2][i+1,j+1] > 0 else self._space[2][i+1,j+1]
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]
        elif i == self.size -1:
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i-1,j-1] = self._space[2][i-1,j-1] - 1/8 if self._space[2][i-1,j-1] > 0 else self._space[2][i-1,j-1]
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
            self._space[2][i-1,j+1] = self._space[2][i-1,j+1] - 1/8 if self._space[2][i-1,j+1] > 0 else self._space[2][i-1,j+1]
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]
        elif j == 0:
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
            self._space[2][i-1,j+1] = self._space[2][i-1,j+1] - 1/8 if self._space[2][i-1,j+1] > 0 else self._space[2][i-1,j+1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
            self._space[2][i+1,j+1] = self._space[2][i+1,j+1] - 1/8 if self._space[2][i+1,j+1] > 0 else self._space[2][i+1,j+1]
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]

        elif j == self.size-1:
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
            self._space[2][i-1,j-1] = self._space[2][i-1,j-1] - 1/8 if self._space[2][i-1,j-1] > 0 else self._space[2][i-1,j-1]
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i+1,j-1] = self._space[2][i+1,j-1] - 1/8 if self._space[2][i+1,j-1] > 0 else self._space[2][i+1,j-1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
        else:
            self._space[2][i,j-1] = self._space[2][i,j-1] - 1/8 if self._space[2][i,j-1] > 0 else self._space[2][i,j-1]
            self._space[2][i,j+1] = self._space[2][i,j+1] - 1/8 if self._space[2][i,j+1] > 0 else self._space[2][i,j+1]
            self._space[2][i-1,j] = self._space[2][i-1,j] - 1/8 if self._space[2][i-1,j] > 0 else self._space[2][i-1,j]
            self._space[2][i-1,j+1] = self._space[2][i-1,j+1] - 1/8 if self._space[2][i-1,j+1] > 0 else self._space[2][i-1,j+1]
            self._space[2][i-1,j-1] = self._space[2][i-1,j-1] - 1/8 if self._space[2][i-1,j-1] > 0 else self._space[2][i-1,j-1]
            self._space[2][i+1,j] = self._space[2][i+1,j] - 1/8 if self._space[2][i+1,j] > 0 else self._space[2][i+1,j]
            self._space[2][i+1,j+1] = self._space[2][i+1,j+1] - 1/8 if self._space[2][i+1,j+1] > 0 else self._space[2][i+1,j+1]
            self._space[2][i+1,j-1] = self._space[2][i+1,j-1] - 1/8 if self._space[2][i+1,j-1] > 0 else self._space[2][i+1,j-1]

    def reset(self):
        state = super().reset()
        self._space[1] = state
        self._space[2] = self.init_conn_layer
        self.actions_history = torch.ones(self.size**2)
        self._step = 0
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        return self._space[1:]

    def step(self, action):
        if self.actions_history[int(action.item())] == 0:
            raise("Took Invalid action!!")
        position = self.action_map[int(action.item())]
        self._space[0][position] = -1
        self._space[1][position] = 1
        self.mark_connectivity_layer(position)
        self.marked +=1
        self.actions_history[int(action.item())] = 0
        s = self._space[1:]
        done = False
        if self._step == self.get_episode_len()-1:
            if self.size < 20:
                firegrid = self.up_scale(self._space[0], 20)
            else:
                firegrid = self._space[0]
            self.write_firewall(firegrid)
            reward = self.generate_reward(self.n_sims_final) 
            done = True
            self._step = 0
            return s, torch.Tensor([reward]).reshape((1, 1)), done
        self._step += 1
        return s, torch.zeros(1), done
