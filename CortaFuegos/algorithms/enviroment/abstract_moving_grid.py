from abc import abstractmethod
from enviroment.abstract_env import Env
import itertools
import numpy as np
import torch
import rasterio
from rasterio.enums import Resampling
import torchvision
import torchvision.transforms.functional as F
import os
class Moving_Grid(Env):
    def __init__(self, size, burn_value = 10, agent_id = -1, agent_dim = 2, n_sims_final = 50, env_id = 0):
        super().__init__(size, burn_value, env_id)
        assert size%2 == 0
        self.name = "moving_grid"
        self.agent_dim = agent_dim # Dimensión del scope del agente
        self.agent_id = agent_id # Número que representa las celdas en donde está el agente
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._space = torch.zeros(3, self.size, self.size).to(self.device) # Privado, corresponde al espacio del ambiente -> Tensor de 3 x size x size
        self._agent_location = np.zeros(2, dtype=int) # Privado, ubicación de la primera entrada del scope del agente 
        self._action_map = {} # Diccionario que mapea el identificador de una acción con la combinación asociada, ej: 0 -> (0, 0, 0, 0)
        combinations = list(itertools.product([0, -1], repeat=agent_dim**2))
        for i in range(2**(agent_dim**2)):
            self._action_map[i] = combinations[i]
        self.n_sims_final = n_sims_final
        # Se incorpora la información correspondiente al tipo de combustible
        absolute_path = os.path.dirname(__file__)
        path = f"{absolute_path}/utils/data/Sub20x20/Forest.asc"
        prop = self.size / 20
        a = self.down_scale(path, prop)
        self._space[2] = self.down_scale(path, prop)/101
        self.forest = self._space[2]
        self.forbidden_cells = int((self.size**2)*0.05)//2
    
    def get_name(self):
        return self.name

    def get_space(self):
        """Función que retorna el espacio del ambiente"""
        return self._space

    def get_episode_len(self):
        return (self.size//2)**2

    def get_agent_location(self):
        """Función que retorna la ubicación del agente"""
        return self._agent_location

    def get_action_map(self):
        """Función que retorna el mapa de acciones"""
        return self._action_map
    
    def get_space_dims(self):
        return (self.size, self.size)

    def get_action_space_dims(self):
        return len(self._action_map)
    
    def get_action_space(self):
        return torch.Tensor([i for i in range(16)]).to(self.device)
    
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
        return torch.Tensor([self._random.randint(0, 15)]).to(self.device)

    def get_action_space(self):
        return torch.Tensor([i for i in range(16)]).to(self.device)

    
    def set_seed(self, seed):
        """Función que establece una semilla para los numeros aleatorios"""
        super().set_seed(seed)

    def reset(self):
        """Función que resetea el ambiente -> Agente en posición (0,0) y toda la grilla sin marcar"""
        self._space[0] = torch.zeros(self.size, self.size).to(self.device)
        self._space[1] = torch.zeros(self.size, self.size).to(self.device)
        self._space[2] = self.forest
        self._agent_location[0] = 0
        self._agent_location[1] = 0
        self.mark_agent()
        return self._space[1:]

    @abstractmethod
    def step(self):
        pass
    
    def sample_space(self):
        """Función que genera un estado aleatorio del sistema"""
        self.reset()
        iterations = self._random.randint(0, (self.size/self.agent_dim)**2)
        for i in range(iterations):
            self.step(self.random_action())
        return self._space[1:]

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
        return torch.Tensor(data).squeeze().to(self.device)
    def up_scale(self, tensor, size):
        t = tensor.unsqueeze(0)
        t_resized = F.resize(t, size, interpolation = torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
        return t_resized.to(self.device)

    def check_if_forbidden(self):
        pos_1 = (self._agent_location[0] < self.forbidden_cells) and (self._agent_location[1] < self.forbidden_cells)
        pos_2 = (self._agent_location[0] < self.forbidden_cells) and (self._agent_location[1] + 1 < self.forbidden_cells)
        pos_3 = (self._agent_location[0] + 1 < self.forbidden_cells) and (self._agent_location[1] < self.forbidden_cells)
        pos_4 = (self._agent_location[0] + 1 < self.forbidden_cells) and (self._agent_location[1] + 1 < self.forbidden_cells)
        mask = torch.ones(16, dtype=torch.bool).to(self.device)
        i = 0
        for j in self._action_map.values():
            if ((j[0] - pos_1 == -2) or (j[1] - pos_2 == -2) or (j[2] - pos_3 == -2) or (j[3] - pos_4 == -2)):
                if j != (0, 0, 0, 0):
                    mask[i] = 0
            i+=1
        return mask
        
    def generate_mask(self):
        return self.check_if_forbidden()
