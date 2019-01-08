import gym
import numpy as np


action_space_len = 3

def greedyAct(_q_dict):
		greedy_act = None
		max_q = -1e10
		for act in range(action_space_len):
			if _q_dict[act] > max_q:
				greedy_act = act
				max_q = _q_dict[act]
		return greedy_act


def epsGreedy(episode, q_dict):
	eps = 1. / episode
	m = action_space_len
	greedy_act = greedyAct(q_dict)
	p = []
	for act in range(action_space_len):
		if act == greedy_act:
			p.append((eps * 1. / m) + 1 - eps)
		else:
			p.append(eps * 1. / m)
	choice = np.random.choice(range(action_space_len), size=1, p=p)
	return choice[0]


def constructQForThisState(_obs, w):
	q_dict = {}
	for i in range(action_space_len):
		q_dict[i] = np.sum(np.append(_obs, i) * w)
	return q_dict


def computeActionValue(_obs, _act, w):
	return np.sum(np.append(_obs, _act) * w)

env = gym.make('MountainCar-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = 3
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# print("action space: ", env.action_space)
# action = env.action_space.sample()
# print("action: ", action)
# print("observation space: ", env.observation_space)
# print(env.observation_space.high)

max_episode = 1
max_step = 200
alpha = 0.5
gamma = 0.9
# Initialize Q
w = np.random.rand(action_space_len)
for i_episode in range(1, max_episode + 1):
	observation = env.reset()  # initialize S
	# Choose action from S using policy dervied from Q
	q_dict = constructQForThisState(observation, w)
	action = epsGreedy(i_episode, q_dict)
	step = 0
	while step  < max_step:
		last_action_value = computeActionValue(observation, action, w)
		last_feature_vector = np.append(observation, action)
		# take action, observe new state & reward
		print("[Episode %d] Action: %d" % (i_episode, action))
		observation, reward, done, info = env.step(action)
		print("[Episode %d] observation: " % i_episode, observation)
		# choose new action
		q_dict = constructQForThisState(observation, w)
		action = epsGreedy(i_episode, q_dict)
		# update Q
		delta_w = alpha * (reward + gamma * computeActionValue(observation, action, w) - last_action_value) *\
					 last_feature_vector 
		w += delta_w
		if done:
			break
		else:
			step += 1

