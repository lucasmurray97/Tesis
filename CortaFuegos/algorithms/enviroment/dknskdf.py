import numpy as np
import torch
ascii_grid = np.loadtxt("utils/data/Sub20x20/Forest.asc", skiprows=6)
space = torch.zeros(3, 20, 20)
space[2] = torch.Tensor(ascii_grid)
print(space)
def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix
print(normalize_2d(ascii_grid))