import gym
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pylab as pl


np.random.seed(4)


def generateCircle(centroid, r, num=50):
	# centroid = (x, y)
	assert len(centroid) == 2
	points_mat = np.zeros((num, 2))
	for i, angle in enumerate(np.linspace(0, 2 * np.pi, num=num)):
		x = centroid[0] + r * np.cos(angle)
		y = centroid[1] + r * np.sin(angle)
		points_mat[i, :] = [x, y]
	return points_mat


action_space_len = 3
env = gym.envs.make("MountainCar-v0")
obs_high = env.observation_space.high
obs_low = env.observation_space.low
receptive_field = 0.1
print("Obs high: ", obs_high)
print("Obs low: ", obs_low)

num_obs_sample = 20  # number of observation sampled on each axis of the observation space
num_features = num_obs_sample**2  # number of features (i.e. len of features vector)

# initialize circles centroid
# centroids_mat = np.zeros((num_obs_sample, 2))  # raw version
# for i in range(2):
# 	# centroids_mat[:, i] = np.linspace(obs_low[i], obs_high[i], num=num_obs_sample)
# 	centroids_mat[:, i] = np.random.rand(num_obs_sample) * (obs_high[i] - obs_low[i]) + obs_low[i]

# centroids_xx, centroids_yy = np.meshgrid(centroids_mat[:, 0], centroids_mat[:, 1])

# centroids_xx[i, j] is paired with centroids_yy[i, j] to create 1 centroid
centroids_list = []
for i in range(num_obs_sample):
	for j in range(num_obs_sample):
		x = np.random.rand() * (obs_high[0] - obs_low[0]) + obs_low[0]
		y = np.random.rand() * (obs_high[1] - obs_low[1]) + obs_low[1]
		centroids_list.append((x, y))
		# centroids_list.append((centroids_xx[i, j], centroids_yy[i, j])) 


# observation = (np.random.rand() * (obs_high[0] - obs_low[0]) + obs_low[0],
# 				 np.random.rand() * (obs_high[1] - obs_low[1]) + obs_low[1])
# feature_vect = obs2FeaturesVector(observation, centroids_list, receptive_field, num_features)
# print("feature_vect: ", feature_vect)

# plt.figure()
# plt.plot([observation[0]], [observation[1]], marker='*', markersize=8)
# for i in range(num_features):
# 	if feature_vect[i] > -1:
# 		centroid = centroids_list[i]
# 		points_mat = generateCircle(centroid, receptive_field)
# 		plt.plot(points_mat[:, 0], points_mat[:, 1])

# plt.xlim(obs_low[0], obs_high[0])
# plt.ylim(obs_low[1], obs_high[1])
# plt.show()


def greedyAct(action_value_list):
	return np.argmax(action_value_list)


def epsilonGreedy(i_episode, action_value_list):
	epsilon = 1. / i_episode
	m = len(action_value_list)
	greedy_act = greedyAct(action_value_list)
	p = []  # probability of being chosen associated with each action
	for act in range(m):
		if act == greedy_act:
			p.append((epsilon * 1. / m) + 1 - epsilon)
		else:
			p.append(epsilon * 1. / m)
	choice = np.random.choice(range(action_space_len), size=1, p=p)
	return choice[0], p


def calActionValue(features_vect, action_value_weights):
	action_value_list = []
	for act in action_value_weights.keys():
		action_value_list.append(np.sum(features_vect * action_value_weights[act]))
	return action_value_list


def obs2FeaturesVector(observation, centroids_list=centroids_list, r=receptive_field, num_features=num_features):
	feat_vect = np.zeros(num_features)
	for centroid in centroids_list:
		dist = np.sqrt((observation[0] - centroid[0])**2 + (observation[1] - centroid[1])**2)
		if dist < r:
			feat_vect[i] = 1
	return feat_vect


def osbervation2Action(observation):
	# build features vector
	features_vect = obs2FeaturesVector(observation)
	# calculate action value for this observation
	action_value_list = calActionValue(features_vect, action_value_weights)
	return epsilonGreedy(i_episode, action_value_list)

#=====================================================#

# Initialize Q(s, a)
action_value_weights = {}
for act in range(action_space_len):
	action_value_weights[act] = np.zeros(num_features)


max_episode = 200
env._max_episode_steps = 10000
steps_per_episode = []
alpha = 0.5/8.
gamma = 0.9
for i_episode in range(1, max_episode + 1):
	observation = env.reset()
	action, p = osbervation2Action(observation)
	done = False
	step = 0
	while not done:
		# env.render()
		# calculate Q(S, A)
		features_vect = obs2FeaturesVector(observation)
		q_S_A = np.sum(action_value_weights[action] * features_vect)
		# take A, observe R & Sprime
		observation_prime, reward, done, info = env.step(action)
		if observation[0] >= 0.1:
			reward = 1.0
		else:
			reward = -1.0
		if reward >= 0:
			print("[Episode %d] get positive reward !!!" % i_episode)
		if done:
			action_value_weights[action] += alpha * (reward - q_S_A) * features_vect
			print("[Episode %d] position: %f\t reward: %f" %(i_episode, observation[0], reward))
			break
		# choose new action
		action_prime, p = osbervation2Action(observation_prime)
		# print("[Episode %d] action_prime: %d" % (i_episode, action_prime))
		# Update Q
		q_Sprime_Aprime = np.sum(action_value_weights[action_prime] * obs2FeaturesVector(observation_prime))
		# if step % 500 == 0:
			# print("[Episode %d] q_S_A = %f\tq_Sprime_Aprime = %f\tr = %f"  %(i_episode, q_S_A, q_Sprime_Aprime, reward))
		target = reward + gamma * q_Sprime_Aprime
		action_value_weights[action] += alpha * (target - q_S_A) * features_vect
		# Move on to new state
		observation = observation_prime
		action = action_prime
		step += 1

	steps_per_episode.append(step)
	print("[Episode %d] finished after %d step." % (i_episode, step))

plt.figure(1)
plt.plot(steps_per_episode)
plt.show()

'''
# Junk !!!

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
# print(env.observation_space.low)

def plotValueFunc(w, fig_index=1):
	fig = plt.figure(fig_index)
	ax = fig.gca(projection='3d')
	x1_vect = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=100)
	x2_vect = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=100)
	x1, x2 = np.meshgrid(x1_vect, x2_vect)
	state_value = np.zeros(x1.shape)
	for i in range(x1.shape[0]):
		for j in range(x1.shape[1]):
			feature = np.array((x1[i, j], x2[i, j]))
			q_dict = constructQForThisState(np.array(feature), w)
			act = greedyAct(q_dict)
			state_value[i, j] = computeActionValue(feature, act, w)

	surf = ax.plot_surface(x1, x2, state_value, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.show()

env = gym.envs.make("MountainCar-v0")
env._max_episode_steps = 4000
max_episode = 100
max_step = 2000
alpha = 0.15
gamma = 0.99
# Initialize Q
w = np.random.rand(action_space_len)
for i_episode in range(1, max_episode + 1):
	observation = env.reset()  # initialize S
	# Choose action from S using policy dervied from Q
	q_dict = constructQForThisState(observation, w)
	action = greedyAct(q_dict)
	step = 0
	if i_episode % 500 == 0:
		render = True
	else:
		render = True

	while step  < max_step:
		if render:
			env.render()
		last_action_value = computeActionValue(observation, action, w)
		last_feature_vector = np.append(observation, action)
		# take action, observe new state & reward
		observation, reward, done, info = env.step(action)
		if reward > 0 :
			print("[Episode %d] [Step %d] Act: %d" % (i_episode, step, action), " obs:", observation, " reward:", reward)		
		# choose new action
		q_dict = constructQForThisState(observation, w)
		action = greedyAct(q_dict)
		# update Q
		loss = reward + gamma * computeActionValue(observation, action, w) - last_action_value
		delta_w = alpha * (reward + gamma * computeActionValue(observation, action, w) - last_action_value) *\
					 last_feature_vector 
		w += delta_w
		if step % 100 == 0:
			print("[Episode %d] [Step %d] Loss: %.4f" % (i_episode, step, loss**2))

		if done:
			print("[Episode %d] finished after %d step." % (i_episode, step))
			break
		else:
			step += 1

	# plot state-value
	if i_episode % 50 == 0:
		plotValueFunc(w, i_episode)
'''
	
env.close()
