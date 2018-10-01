'''
minimum example
source: https://gym.openai.com/docs/#installation
'''

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(200):
	env.render()
	env.step(env.action_space.sample()) # take a random action

env.close()


'''
Original:
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''