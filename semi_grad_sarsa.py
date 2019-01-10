import numpy as np
import gym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import pickle


env = gym.envs.make("MountainCar-v0")
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
n_components = 25
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])
featurizer.fit(scaler.transform(observation_examples))


def createFeatures(state):
	scaled = scaler.transform([state])
	featurized = featurizer.transform(scaled)
	return featurized[0]


def greedy(action_value_list):
	return np.argmax(action_value_list)


def epsilonGreedy(action_value_list, epsilon):
	m = len(action_value_list)
	greedy_act = greedy(action_value_list)
	prob = []  # probability of being chosen associated with each action
	for act in range(m):
		if act == greedy_act:
			prob.append((epsilon * 1. / m) + 1 - epsilon)
		else:
			prob.append(epsilon * 1. / m)
	choice = np.random.choice(range(m), size=1, p=prob)
	return choice[0], prob


def calQ(state, action, w):
	features = createFeatures(state)
	return np.dot(features, w[action])


def return_decayed_value(starting_value, minimum_value, global_step, decay_step):
    """Returns the decayed value.
    decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
    @param starting_value the value before decaying
    @param global_step the global step to use for decay (positive integer)
    @param decay_step the step at which the value is decayed
    """
    decayed_value = starting_value * np.power(0.9, (global_step*1./decay_step))
    if decayed_value < minimum_value:
        return minimum_value
    else:
        return decayed_value


def plotValueFunc(w, max_episode, epsilon):
	x_vect = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=50)
	xdot_vect = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=50)
	x, xdot = np.meshgrid(x_vect, xdot_vect)
	state_value = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			state = (x[i, j], xdot[i, j])
			action_value_list = [calQ(state, i, w) for i in range(env.action_space.n)]
			act, p = epsilonGreedy(action_value_list, epsilon)
			for act in range(env.action_space.n):	
				state_value[i, j] += p[act] * action_value_list[act]
	# plot Value function
	fig = plt.figure(max_episode)
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(x, xdot, -state_value, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Value')
	ax.set_title("Mountain \"Cost To Go\" Function")
	plt.show()


class ReplayMemory(object):
	"""docstring for ReplayMemory"""
	def __init__(self, memory_capacity=500):
		self.data = []  # FIFO, IN at top, OUT at bottom
		self.capacity = memory_capacity

	def sample(self, num_sample=5):
		return [self.data[i] for i in np.random.randint(0, high=len(self.data), size=num_sample)]

	def add(self, state, action, reward, new_state):
		if len(self.data) == self.capacity:
			self.data.pop()
		self.data.insert(0, (state, action, reward, new_state))



env._max_episode_steps = 2000
obs_high = env.observation_space.high
obs_low = env.observation_space.low


num_features = 4 * n_components

gamma = 0.99
alpha = 0.001

epsilon_start = 0.9  # those are the values for epsilon decay
epsilon_stop = 0.01
epsilon_decay_step = 10

max_episode = 50
episode_switch = 10  # after 10 episode, the roles of 2 set of weights are switch
steps_per_episode = []
epsilon = epsilon_start

memory = ReplayMemory()  # init memory for experience replay

# init 2 set of weights to serve fixed Q-target
weights = {}
for act in range(env.action_space.n):
	weights[act] = np.zeros(num_features)
weights_target = weights

flag_froze_1 = True
for i_episode in range(1, max_episode + 1):
	# Change role of 2 set of weights
	if i_episode % episode_switch == 0:
		weights_target = weights

	epsilon = return_decayed_value(epsilon_start, epsilon_stop, i_episode, epsilon_decay_step)
	# epsilon = 1./i_episode
	
	# initialzie state & action of episode
	state = env.reset()
	action_value_list = [calQ(state, i, weights) for i in range(env.action_space.n)]
	action, prob = epsilonGreedy(action_value_list, epsilon)
	step = 0
	done = False
	while not done:
		features = createFeatures(state)  # create features vector for current state

		new_state, reward, done, _info = env.step(action)

		# update memory
		memory.add(state, action, reward, new_state)

		if new_state[0] > 0.45:
			done = True
			reward = 1.0

		if done:
			weights[action] += alpha * (reward - calQ(state, action, weights)) * features
			print("Episode %d finishes in %d steps." % (i_episode, step))
			break
		
		# Experience replay
		for exp in memory.sample():
			_state = exp[0]
			_action = exp[1]
			_reward = exp[2]
			_new_state = exp[3]
			_features = createFeatures(_state)
			# choose new action
			_action_value_list = [calQ(_new_state, i, weights) for i in range(env.action_space.n)]
			_new_action, prob = epsilonGreedy(_action_value_list, epsilon)
			# Update action-value function
			weights[_action] += alpha * (_reward + gamma * calQ(_new_state, _new_action, weights_target) - calQ(_state, _action, weights)) * _features

		# move on
		state = new_state
		action = new_action
		step += 1

	steps_per_episode.append(step)

# store weights
# with open('mountain_car_weights.p', 'wb') as fp:
    # pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure(1)
plt.plot(steps_per_episode)
plt.xlabel("Episode")
plt.ylabel("Number of steps")
plt.show()

plotValueFunc(weights, max_episode, epsilon)


