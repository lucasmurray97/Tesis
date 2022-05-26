import itertools
import numpy as np
class FireGrid:
    def __init__(self, size, agent_id = 2, agent_dim = 2):
        assert size%2 == 0
        self.size = size
        self.agent_dim = agent_dim
        self.agent_id = agent_id
        self._space = np.zeros((self.size, self.size), dtype=int)
        self._agent_location = np.zeros(2, dtype=int)
        self._action_map = {}
        combinations = list(itertools.product([0, 1], repeat=agent_dim**2))
        for i in range(2**(agent_dim**2)):
            self._action_map[i] = combinations[i]

    def get_space(self):
        return self._space

    def get_agent_location(self):
        return self._agent_location
    
    def get_action_map(self):
        return self._action_map

    def reset(self):
        self._space = np.zeros((self.size, self.size), dtype=int)
        self._agent_location[0] = 0
        self._agent_location[1] = 0
        self._space[self._agent_location[0], self._agent_location[1]] = 2
        self._space[self._agent_location[0], self._agent_location[1] + 1] = 2
        self._space[self._agent_location[0] +1, self._agent_location[1]] = 2
        self._space[self._agent_location[0] + 1, self._agent_location[1] + 1] = 2
        return self._space

    def step(self, action):
        action_tuple = self._action_map[action]
        print(self._agent_location)
        self._space[self._agent_location[0], self._agent_location[1]] = action_tuple[0]
        self._space[self._agent_location[0], self._agent_location[1] + 1] = action_tuple[1]
        self._space[self._agent_location[0] +1, self._agent_location[1]] = action_tuple[2]
        self._space[self._agent_location[0] + 1, self._agent_location[1] + 1] = action_tuple[3]
        if self._agent_location[0] < self.size - self.agent_dim:
            self._agent_location[0] += self.agent_dim
        elif self._agent_location[1] < self.size - self.agent_dim:
            self._agent_location[0] = 0
            self._agent_location[1] += self.agent_dim
        else:
            final_reward = -20
            return self._space, True, final_reward
        self._space[self._agent_location[0], self._agent_location[1]] = 2
        self._space[self._agent_location[0], self._agent_location[1] + 1] = 2
        self._space[self._agent_location[0] +1, self._agent_location[1]] = 2
        self._space[self._agent_location[0] + 1, self._agent_location[1] + 1] = 2
        return self._space, -1, False
# env = FireGrid(20)
# print(env.reset())
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.step(15))
# print(env.get_space())
# print(env.step(4))
# print(env.get_space())
# print(env.reset())

