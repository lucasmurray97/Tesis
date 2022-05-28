from firegrid import FireGrid
env = FireGrid(20)
env.reset()
# env.show_state()
for i in range(100):
    state, r, done = env.step(env.random_action())
    # env.show_state()
print(done)
print(r)


