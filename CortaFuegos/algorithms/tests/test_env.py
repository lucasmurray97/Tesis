from enviroment.firegrid import FireGrid
import numpy as np

def test_reset():
    env = FireGrid(20)
    space = env.reset()
    example = np.zeros((20, 20), dtype=int)
    example[0,0] = 1
    example[0,1] = 1
    example[1,0] = 1
    example[1,1] = 1
    assert np.array_equal(space,example) == True

def test_step():
    env = FireGrid(20)
    space = env.reset()
    space, r, done = env.step(0)
    example = np.zeros((20, 20), dtype=int)
    example[2,0] = 1
    example[2,1] = 1
    example[3,0] = 1
    example[3,1] = 1
    assert np.array_equal(space,example) == True
    assert done == False
    assert r == -1

def test_done():
    env = FireGrid(20)
    env.reset()
    # env.show_state()
    for i in range(100):
        state, r, done = env.step(env.random_action())
    assert done == True


