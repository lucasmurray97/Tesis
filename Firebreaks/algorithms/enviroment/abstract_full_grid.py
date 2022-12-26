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
class Abstract_Full_Grid(Env):
    def __init__(self, size, burn_value = 10, n_sims_final = 50, env_id = 0):
        super().__init__(size, burn_value, env_id)
        assert size%2 == 0
        self.n_sims_final = n_sims_final
        self.name = "full_grid"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._space = torch.zeros(2, self.size, self.size)
        self.action_map = {}
        self.forbidden_cells = []
        pos = 0
        for i in range(self.size):
            for j in range(self.size):
                self.action_map[pos] = (i,j)
                if i < int((self.size**2)*0.05)//2:
                    if j < int((self.size**2)*0.05)//2:
                        self.forbidden_cells.append(pos)
                pos += 1
        # Se incorpora la información correspondiente al tipo de combustible
        absolute_path = os.path.dirname(__file__)
        path = f"{absolute_path}/utils/data/Sub20x20_{self.env_id}/Forest.asc"
        prop = self.size / 20
        a = self.down_scale(path, prop)
        forest = self.down_scale(path, prop)/101
        self._space[1] = forest
        self.forest = forest

    def get_name(self):
        return self.name

    def get_space(self):
        """Función que retorna el espacio del ambiente"""
        return self._space

    def get_episode_len(self):
        """5% del total de celdas"""
        quant = int((self.size**2)*0.05)
        if quant%2 == 0:
            quant += 1
        return quant

    def get_action_map(self):
        """Función que retorna el mapa de acciones"""
        return self.action_map
    
    def get_space_dims(self):
        return (self.size, self.size)

    def get_action_space_dims(self):
        return len(self.action_map)
    
    def random_action(self):
        """Función que retorna una acción aleatoria del espacio de acciones"""
        return torch.Tensor([self._random.randint(0, self.size**2 - 1)]).to(self.device)

    def get_action_space(self):
        return torch.Tensor([i for i in range(self.size**2)])

    def set_seed(self, seed):
        """Función que establece una semilla para los numeros aleatorios"""
        super().set_seed(seed)

    def reset(self):
        self._space[0] = torch.zeros(self.size, self.size)
        self._space[1] = self.forest
        return self._space[1].unsqueeze(0)

    @abstractmethod
    def step(self):
        pass
    
    def sample_space(self):
        """Función que genera un estado aleatorio del sistema"""
        self.reset()
        iterations = self._random.randint(0, self.get_episode_len())
        for i in range(iterations):
            self.step(self.random_action())
        return self._space[1:]

    def show_state(self):
        """Función que printea el estado del ambiente"""
        print(f"Estado de la grilla del ficticia:")
        print(self._space[0])
        print(f"Estado de la grilla del bosque:")
        print(self._space[1])
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
        return torch.Tensor(data).squeeze()
    def up_scale(self, tensor, size):
        t = tensor.unsqueeze(0)
        t_resized = F.resize(t, size, interpolation = torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
        return t_resized
