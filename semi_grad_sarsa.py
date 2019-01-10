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

featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
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


env._max_episode_steps = 2000
obs_high = env.observation_space.high
obs_low = env.observation_space.low


num_features = 400

gamma = 0.99
alpha = 0.001
epsilon_start = 0.9  # those are the values for epsilon decay
epsilon_stop = 0.01
epsilon_decay_step = 10
max_episode = 200

steps_per_episode = []
epsilon = epsilon_start

weights = {}
for act in range(env.action_space.n):
	weights[act] = np.zeros(num_features)

for i_episode in range(1, max_episode + 1):
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

		if new_state[0] > 0.45:
			done = True
			reward = 1.0

		if done:
			weights[action] += alpha * (reward - calQ(state, action, weights)) * features
			print("Episode %d finishes in %d steps." % (i_episode, step))
			break
		# choose new action
		action_value_list = [calQ(new_state, i, weights) for i in range(3)]
		new_action, prob = epsilonGreedy(action_value_list, epsilon)
		weights[action] += alpha * (reward + gamma * calQ(new_state, new_action, weights) - calQ(state, action, weights)) * features

		# move on
		state = new_state
		action = new_action
		step += 1

	steps_per_episode.append(step)

# store weights
with open('mountain_car_weights.p', 'wb') as fp:
    pickle.dump(weights, fp, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure(1)
plt.plot(steps_per_episode)
plt.xlabel("Episode")
plt.ylabel("Number of steps")
plt.show()

plotValueFunc(weights, max_episode, epsilon)


