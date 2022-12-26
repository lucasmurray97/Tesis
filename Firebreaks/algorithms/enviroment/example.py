from moving_grid.moving_gym_v6 import Moving_Gym_v6
env = Moving_Gym_v6(20)
state = env.reset()
print(state)
# env.show_state()
# for i in range(100):
#     env.step(env.random_action())
#     env.show_state()