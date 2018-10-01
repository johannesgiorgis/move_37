'''
better example
source: https://github.com/openai/gym/issues/430
'''

import gym
from gym import wrappers
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, path)

for i_episode in range(20):
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