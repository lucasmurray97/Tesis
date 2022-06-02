from enviroment.firegrid import FireGrid
import time
env = FireGrid(20)
start = time.time()
for i in range(1):
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
print(state)
end = time.time()
print(f"Test took {end-start} secs")
