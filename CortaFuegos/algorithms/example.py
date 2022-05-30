from enviroment.firegrid import FireGrid
import time
env = FireGrid(20)
start = time.time()
for i in range(10):
    env.reset()
    for i in range(100):
        state, r, done = env.step(env.random_action())
end = time.time()
print(f"Test took {end-start} secs")
