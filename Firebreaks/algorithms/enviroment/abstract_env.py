from abc import ABC, abstractmethod
from enviroment.utils.final_reward import write_firewall_file, generate_reward 
import random
import os
class Env(ABC):
    def __init__(self, size, burn_value = 10, env_id = 0, instance = "homo_1"):
        self.size = size
        self.burn_value = burn_value
        self._random = random
        self.env_id = env_id
        self.instance = instance

    @abstractmethod
    def get_space(self):
        pass

    @abstractmethod
    def get_episode_len(self):
        pass

    @abstractmethod
    def get_space(self):
        pass
    @abstractmethod
    def get_action_space(self):
        pass
    @abstractmethod
    def get_space_dims(self):
        pass
    @abstractmethod
    def get_action_space_dims(self):
        pass
    @abstractmethod
    def random_action(self):
        pass
    
    def set_seed(self, seed):
        """Función que establece una semilla para los numeros aleatorios"""
        self._random.seed(seed)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def sample_space(self):
        pass

    @abstractmethod
    def show_state(self):
        pass
    
    def write_firewall(self, firegrid): 
        write_firewall_file(firegrid, self.env_id, self.instance)
    
    def generate_reward(self, sims):
        return generate_reward(sims, self.size, self.env_id, self.instance)*self.burn_value
