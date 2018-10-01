'''
Simple Reinforcement Learning Methods to learn CartPole
'''
import gym
import numpy as np


def run_episode(env, parameters, should_render=False):
	observation = env.reset()
	total_reward = 0
	for _ in range(200):
		if should_render:
			env.render()
		action = 0 if np.matmul(parameters, observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	#env.close()
	return total_reward


def random_search(env):
	'''
	random search algorithm
	try random weights and pick the one that performs the best
	'''
	best_params = None
	best_reward = 0
	for i_episode in range(1000):
		parameters = np.random.rand(4) * 2 - 1
		reward = run_episode(env, parameters, should_render=False)
		#print(f"Reward is {reward}")
		if reward > best_reward:
			best_reward = reward
			best_params = parameters
			# considered solved if the agent lasts 200 timesteps
			if reward == 200:
				#print(f"It took {i_episode} episodes")
				break
	return i_episode


def hill_climb(env):
	'''
	hill-climbing algorithm
	Start with some randomly chosen initial weights. At every episode,
	we add some noise to the weights, and keep the new weights if the 
	agent improves
	'''
	noise_scaling = 0.5
	parameters = np.random.rand(4) * 2 - 1
	best_reward = 0
	for i_episode in range(10000):
		new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
		reward = 0
		reward = run_episode(env, new_params)
		#print(f"Reward is {reward}")
		if reward > best_reward:
			best_reward = reward
			parameters = new_params
			# considered solved if the agent lasts 200 timesteps
			if reward == 200:
				#print(f"It took {i_episode} episodes")
				break
	return i_episode


def better_hill_climb(env):
	'''
	hill-climbing algorithm
	Start with some randomly chosen initial weights. At every episode,
	we add some noise to the weights, and keep the new weights if the 
	agent improves
	Better: Instead of running one episode to measure how good a set of
	weights is, we run it multiple times and sum up the rewards
	'''
	noise_scaling = 0.1
	parameters = np.random.rand(4) * 2 - 1
	best_reward = 0
	for i_episode in range(10000):
		new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
		reward = 0
		episodes_per_update = 2
		for _ in range(episodes_per_update):
			run = run_episode(env, new_params)
			reward += run
		#print(f"Reward is {reward}")
		if reward > best_reward:
			best_reward = reward
			parameters = new_params
			# considered solved if the agent lasts 200 timesteps
			if reward == 200:
				print(f"It took {i_episode} episodes")
				break
	return i_episode


def run_rl_algorithm(env, rl_func, iterations):
	'''
	run specified rl algorithm for specified iterations
	'''
	results = []
	print(f"Running {rl_func.__name__} algorithm for {iterations} iterations...")
	for i in range(iterations):
		episode_taken = rl_func(env)
		print(f"{i} - It took {episode_taken} episodes")
		results.append(episode_taken)

	#print(f"\nResult: {results}")
	#print("\n")
	average_episode = sum(results) / len(results)
	#print(f"Average episode: {average_episode} episodes")
	print("Average Episode|Results")
	print(f"{average_episode} | {results}")



# MAIN
if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	run_rl_algorithm(env, random_search, 100)
	#run_rl_algorithm(env, hill_climb, 10)
	#run_rl_algorithm(env, better_hill_climb, 2)
	env.close()