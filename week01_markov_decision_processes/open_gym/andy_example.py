'''
andy's example
'''

import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):
    print(f"i is {i_episode}")
    observation = env.reset()
    t = 0
    while True:
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        t += 1
        if done:
           print("Done after {} steps".format(t+1))
           break

env.close()