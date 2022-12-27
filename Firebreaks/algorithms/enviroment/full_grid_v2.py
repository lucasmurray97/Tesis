import torch
import torch.nn.functional as F
from enviroment.abstract_full_grid import Abstract_Full_Grid
import torch.nn.functional as F
import numpy as np
class Full_Grid_V2(Abstract_Full_Grid):
    def __init__(self, size, burn_value = 10, n_sims_final = 50, env_id = 0):
        super().__init__(size, burn_value, n_sims_final, env_id)
        self.actions_history = torch.ones(self.size**2, dtype=torch.bool)
        self.shape = (2, self.size**2, self.size**2)
        self.initial_matrix = self.generate_matrix()
        self.adjacency_matrix = self.initial_matrix
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        self._step = 0
        self.marked = 0

    def generate_matrix(self):
        N = self.size**2
        """Creates a 2D grid adjacency matrix."""
        sqN = np.sqrt(N).astype(int)  # There will sqN many nodes on x and y
        adj = np.zeros((sqN, sqN, sqN, sqN), dtype=bool)
        # Take adj to encode (x,y) coordinate to (x,y) coordinate edges
        # Let's now connect the nodes
        for i in range(sqN):
            for j in range(sqN):
                # Connect x=i, y=j, to x-1 and x+1, y-1 and y+1
                adj[i, j, max((i-1), 0):(i+2), max((j-1), 0):(j+2)] = True
                # max is used to avoid negative slicing, and +2 is used because
                # slicing does not include last element.
        adj = adj.reshape(N, N)  # Back to node-to-node shape
        # Remove self-connections (optional)
        adj ^= np.eye(N, dtype=bool)
        adj_tensor = torch.from_numpy(adj.astype(int))
        return adj_tensor


    # def pad_state(self, state):
    #     padding = (self.size**2)//2 - self.size//2
    #     padded_state = F.pad(state.squeeze(0), (padding,padding,padding,padding), "constant", 0.)
    #     return padded_state

    def mark_matrix(self, action):
        pos = int(action.item())
        for i in range(self.size**2):
            self.adjacency_matrix[pos][i] = 0.
            self.adjacency_matrix[i][pos] = 0.

    def reset(self):
        state = super().reset()
        self.actions_history = torch.ones(self.size**2)
        self._step = 0
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        self.adjacency_matrix = self.initial_matrix
        upped_state = self.up_scale(state, self.size**2)
        stacked_state = torch.stack((upped_state.squeeze(0), self.adjacency_matrix))
        return stacked_state

    def step(self, action):
        position = self.action_map[int(action.item())]
        self._space[0][position] = -1
        self._space[1][position] = 1
        self.mark_matrix(action)
        self.marked +=1
        self.actions_history[int(action.item())] = 0
        s = self._space[1]
        upped_state = self.up_scale(s, self.size**2)
        state = torch.stack((upped_state, self.adjacency_matrix))
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
            return state, torch.Tensor([reward]).reshape((1, 1)), done
        self._step += 1
        return state, torch.zeros(1), done
