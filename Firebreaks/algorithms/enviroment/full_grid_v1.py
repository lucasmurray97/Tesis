import torch
import torch.nn.functional as F
from enviroment.abstract_full_grid import Abstract_Full_Grid

class Full_Grid_V1(Abstract_Full_Grid):
    def __init__(self, size, burn_value = 10, n_sims_final = 50, env_id = 0, instance = "homo_1", gpu=False):
        super().__init__(size, burn_value, n_sims_final, env_id, instance)
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        self._step = 0
        self.marked = 0
        self.shape = (1, self.size, self.size)

    def reset(self):
        state = super().reset()
        self.actions_history = torch.ones(self.size**2)
        self._step = 0
        for i in self.forbidden_cells:
            self.actions_history[i] = 0
        return state

    def step(self, action):
        if self.actions_history[int(action.item())] == 0:
            raise("Took Invalid action!!")
        position = self.action_map[int(action.item())]
        self._space[0][position] = -1
        self._space[1][position] = 1
        self.marked +=1
        self.actions_history[int(action.item())] = 0
        s = self._space[1].unsqueeze(0)
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
