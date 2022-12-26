import gym
from gym import spaces
import itertools
import torch
from utils.final_reward import write_firewall_file, generate_reward 
import rasterio
from rasterio.enums import Resampling
import torchvision
import torchvision.transforms.functional as F
class Abstract_Moving_Gym(gym.Env):
    metadata = {"render_modes": [None], "render_fps": None}
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10, n_sims_final = 10, env_id = 0):
        self.size = size
        self.burn_value = burn_value
        self.n_sims_final = n_sims_final
        self.agent_id = agent_id
        self.agent_dim = agent_dim
        self._agent_location = torch.zeros(2, dtype=int) # Privado, ubicación de la primera entrada del scope del agente 
        self.firegrid = torch.zeros(self.size, self.size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, self.size, self.size))
        self.action_space = spaces.Discrete(16)
        self._action_map = {} # Diccionario que mapea el identificador de una acción con la combinación asociada, ej: 0 -> (0, 0, 0, 0)
        combinations = list(itertools.product([0, -1], repeat=agent_dim**2))
        for i in range(2**(agent_dim**2)):
            self._action_map[i] = combinations[i]
        path = "/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/moving_grid/utils/data/Sub20x20_0/Forest.asc"
        prop = self.size / 20
        a = self.down_scale(path, prop)
        self.observation_space[1] = self.down_scale(path, prop)/101
        self.forest = self.observation_space[1]
        self.forbidden_cells = int((self.size**2)*0.05)//2
        self.window = None
        self.clock = None
        self.env_id = 0
    
    def mark_agent(self):
        """Función que dada la ubicación del agente, marca esta en la grilla"""
        self.observation_space[0][self._agent_location[0], self._agent_location[1]] = 1
        self.observation_space[0][self._agent_location[0], self._agent_location[1] + 1] = 1
        self.observation_space[0][self._agent_location[0] +1, self._agent_location[1]] = 1
        self.observation_space[0][self._agent_location[0] + 1, self._agent_location[1] + 1] = 1
    
    def erase_agent(self):
        """Función que dada la ubicación del agente, borra el agente de la grilla"""
        self.observation_space[0][self._agent_location[0], self._agent_location[1]] = 0
        self.observation_space[0][self._agent_location[0], self._agent_location[1] + 1] = 0
        self.observation_space[0][self._agent_location[0] +1, self._agent_location[1]] = 0
        self.observation_space[0][self._agent_location[0] + 1, self._agent_location[1] + 1] = 0

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)
        self.firegrid = torch.zeros(self.size, self.size)
        self.observation_space[0] = torch.zeros(self.size, self.size)
        self.observation_space[1] = self.forest
        self._agent_location[0] = 0
        self._agent_location[1] = 0
        self.mark_agent()
        return self.observation_space

    def step(self, action):
        pass
    def render(self):
        return self._render_frame()
    def _render_frame(self):
        return None
    def write_firewall(self, firegrid): 
        write_firewall_file(firegrid, self.env_id)
    
    def generate_reward(self, sims):
        return generate_reward(sims, self.size, self.env_id)*self.burn_value
    
    def down_scale(self, path, prop):
        with rasterio.open(path) as dataset:
            # resample data to target shape
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * prop),
                    int(dataset.width * prop)
                ),
                resampling=Resampling.mode
            )
        return torch.Tensor(data).squeeze()

    def up_scale(self, tensor, size):
        t = tensor.unsqueeze(0)
        t_resized = F.resize(t, size, interpolation = torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
        return t_resized