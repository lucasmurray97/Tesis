import sys
sys.path.append("/home/lucas/Escritorio/Universidad/Tesis/CortaFuegos/algorithms/")
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
import torch
import numpy as np

def test_reset_v1():
    env = FireGrid(20)
    space = env.reset()
    example = torch.zeros((1, 20, 20))
    example[0][0,0] = 1
    example[0][0,1] = 1
    example[0][1,0] = 1
    example[0][1,1] = 1
    assert torch.equal(space,example) == True

def test_step_v1():
    env = FireGrid(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros((1, 20, 20))
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    assert torch.equal(space,example) == True
    assert done == False
    assert r == torch.Tensor([0])

def test_done_v1():
    env = FireGrid(20)
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
    assert done == True

def test_reset_v3():
    env = FireGrid_V3(20)
    space = env.reset()
    example = torch.zeros(2, 20, 20)
    example[1][0,0] = 1
    example[1][0,1] = 1
    example[1][1,0] = 1
    example[1][1,1] = 1
    assert torch.equal(space,example) == True

def test_step_v3():
    env = FireGrid_V3(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(2, 20, 20)
    example[1][2,0] = 1
    example[1][2,1] = 1
    example[1][3,0] = 1
    example[1][3,1] = 1
    assert torch.equal(space,example) == True
    assert done == False
    assert r == torch.Tensor([0])

def test_done_v3():
    env = FireGrid_V3(20)
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
    assert done == True
