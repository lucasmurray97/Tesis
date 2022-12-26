import sys
sys.path.append("/home/lucas/Tesis/Tesis/CortaFuegos/algorithms/")
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from enviroment.firegrid_v5 import FireGrid_V5
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
import torch
import numpy as np
forest_data_complete = np.loadtxt("/home/lucas/Tesis/CortaFuegos/algorithms/tests/Forest.asc", skiprows=6)
norm = np.linalg.norm(forest_data_complete)
forest_data = forest_data_complete/norm
t_forest_data = torch.Tensor(forest_data)
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

def test_reset_v4():
    env = FireGrid_V4(20)
    space = env.reset()
    example = torch.zeros(3, 20, 20)
    example[1][0,0] = 1
    example[1][0,1] = 1
    example[1][1,0] = 1
    example[1][1,1] = 1
    example[2] = t_forest_data
    assert torch.equal(space,example) == True

def test_step_v4():
    env = FireGrid_V4(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(3, 20, 20)
    example[1][2,0] = 1
    example[1][2,1] = 1
    example[1][3,0] = 1
    example[1][3,1] = 1
    example[2] = t_forest_data
    assert torch.equal(space,example) == True
    assert done == False
    assert r == torch.Tensor([0])

def test_done_v4():
    env = FireGrid_V4(20)
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
    assert done == True

def test_reset_v5():
    env = FireGrid_V5(20)
    space = env.reset()
    example = torch.zeros(3, 20, 20)
    example[1][0,0] = 1
    example[1][0,1] = 1
    example[1][1,0] = 1
    example[1][1,1] = 1
    example[2] = t_forest_data
    assert torch.equal(space,example) == True

def test_step_v5():
    env = FireGrid_V5(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(3, 20, 20)
    example[1][2,0] = 1
    example[1][2,1] = 1
    example[1][3,0] = 1
    example[1][3,1] = 1
    example[2] = t_forest_data
    assert torch.equal(space,example) == True
    assert done == False

def test_done_v5():
    env = FireGrid_V5(20)
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
    assert done == True

def test_reset_v6():
    env = FireGrid_V6(20)
    space = env.reset()
    example = torch.zeros(2, 20, 20)
    example[0][0,0] = 1
    example[0][0,1] = 1
    example[0][1,0] = 1
    example[0][1,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True

def test_step_v6():
    env = FireGrid_V6(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True
    assert done == False

def test_step_2_v6():
    env = FireGrid_V6(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([1]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert example[1][1,1] == 1
    assert torch.equal(space,example) == True
    assert done == False

def test_done_v6():
    env = FireGrid_V6(20)
    env.reset()
    for i in range(100):
        a = env.random_action()
        state, r, done = env.step(a)
    assert done == True

def test_reset_v7():
    env = FireGrid_V7(20)
    space = env.reset()
    example = torch.zeros(2, 20, 20)
    example[0][0,0] = 1
    example[0][0,1] = 1
    example[0][1,0] = 1
    example[0][1,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True

def test_step_v7():
    env = FireGrid_V7(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True
    assert done == False

def test_step_2_v7():
    env = FireGrid_V7(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([1]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert example[1][1,1] == 1
    assert torch.equal(space,example) == True
    assert done == False

def test_done_v7():
    env = FireGrid_V7(20)
    env.reset()
    for i in range(100):
        a = env.random_action()
        state, r, done = env.step(a)
    assert done == True

def test_reset_v8():
    env = FireGrid_V8(20)
    space = env.reset()
    example = torch.zeros(2, 20, 20)
    example[0][0,0] = 1
    example[0][0,1] = 1
    example[0][1,0] = 1
    example[0][1,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True

def test_step_v8():
    env = FireGrid_V8(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([0]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert torch.equal(space,example) == True
    assert done == False

def test_done_v8():
    env = FireGrid_V8(20)
    env.reset()
    for i in range(100):
        a = env.random_action()
        state, r, done = env.step(a)
        state_f = env._space[0].clone()
        assert torch.equal(env.last_complete, state_f)
    assert done == True
    env.reset()
    assert torch.equal(env.last_complete, state_f)

def test_step_2_v8():
    env = FireGrid_V8(20)
    space = env.reset()
    space, r, done = env.step(torch.Tensor([1]))
    example = torch.zeros(2, 20, 20)
    example[0][2,0] = 1
    example[0][2,1] = 1
    example[0][3,0] = 1
    example[0][3,1] = 1
    example[1] = torch.Tensor(forest_data_complete/101)
    assert example[1][1,1] == 1
    assert torch.equal(space,example) == True
    assert done == False