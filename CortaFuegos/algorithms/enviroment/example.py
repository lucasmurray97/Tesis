from firegrid_v8 import FireGrid_V68
env = FireGrid_V8(20)
env.reset()
env.show_state()
for i in range(100):
    env.step(env.random_action())
    env.show_state()