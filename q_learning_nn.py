import torch
import torch.nn.functional as F
import numpy as np
import gym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


dtype = torch.float
device = torch.device("cuda:0")


env = gym.envs.make("MountainCar-v0")
env._max_episode_steps = 2001


def greedy(action_value_list):
	return np.argmax(action_value_list)


def epsilonGreedy(action_value_list, epsilon, outputprob=False):
	m = len(action_value_list)
	greedy_act = greedy(action_value_list)
	prob = []  # probability of being chosen associated with each action
	for act in range(m):
		if act == greedy_act:
			prob.append((epsilon * 1. / m) + 1 - epsilon)
		else:
			prob.append(epsilon * 1. / m)
	choice = np.random.choice(range(m), size=1, p=prob)
	
	if outputprob:
		return choice[0], prob
	else:
		return choice[0]


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


epsilon_start = 0.9  # those are the values for epsilon decay
epsilon_stop = 0.01
epsilon_decay_step = 10
epsilon = epsilon_start
memory = ReplayMemory(1000)
update_target_net = 3
gamma = 0.999

batch_size, n_in, n_hidden1, n_hidden2, n_out = 150, 2, 256, 256, 3  # n_in = dim of obs space, n_out = dim of action space

policy_net = torch.nn.Sequential(
	torch.nn.Linear(n_in, n_hidden1),
	torch.nn.ReLU(),
	torch.nn.Linear(n_hidden1, n_hidden2),
	torch.nn.ReLU(),
	torch.nn.Linear(n_hidden2, n_out)
)

def init_weights(m):
	if type(m) == torch.nn.Linear:
		m.weight.data.normal_(0.0, 0.05)
		m.bias.data.normal_(0.0, 0.05)

policy_net.apply(init_weights)


target_net = torch.nn.Sequential(
	torch.nn.Linear(n_in, n_hidden1),
	torch.nn.ReLU(),
	torch.nn.Linear(n_hidden1, n_hidden2),
	torch.nn.ReLU(),
	torch.nn.Linear(n_hidden2, n_out)
)
target_net.load_state_dict(policy_net.state_dict())

policy_net.to(device)
target_net.to(device)
# loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(policy_net.parameters(), lr = 0.001)
# state = env.observation_space.sample()
# state = torch.tensor(state, device=device, dtype=dtype)
# print("state: ", state)
# print("forward: ", policy_net(state))
# action_value_list = policy_net(state).tolist()
# print("action_value_list: ",action_value_list)

step_per_episode = []
act_dict = {}
for i_episode in range(1,20):
	state = env.reset()
	state = torch.tensor(state.reshape((1, 2)), device=device, dtype=dtype)  # convert state to tensor
	done = False
	step = 0
	# calculate epsilon
	epsilon = return_decayed_value(epsilon_start, epsilon_stop, i_episode, epsilon_decay_step)
	# epsilon = 1./i_episode
	act_dict[i_episode] = []
	while not done:
		# choose A from S using behaviour policy
		action_value_tensor = policy_net(state)
		# print("action_value_list: ", action_value_tensor.data.tolist())
		action = epsilonGreedy(action_value_tensor.data.tolist()[0], epsilon)
		# print("Episode %d action: " %i_episode, action)
		act_dict[i_episode].append(action)
		# take A, observe R and S'
		new_state, reward, done, _ = env.step(action)
		if new_state[0] > 0.45:
			reward = 1.
			new_state[0] = 0
			new_state[1] = 0  # to produce a tensor of all 0 after passed through target net
			done = True

		new_state = torch.tensor(new_state.reshape((1, 2)), device=device, dtype=dtype)
		reward = torch.tensor([[reward]], device=device, dtype=dtype)
		# Update memory
		memory.add(state, action, reward, new_state)

		if len(memory.data) == 1000:
			# Experience Replay
			S_tensor = None
			reward_tensor = None
			new_S_tensor = None
			action_arr = np.zeros((batch_size, 3))
			for i, exp in enumerate(memory.sample(batch_size)):
				if i == 0:
					S_tensor = exp[0].clone()
					reward_tensor = exp[2].clone()
					new_S_tensor = exp[3].clone()
				else:
					S_tensor = torch.cat((S_tensor, exp[0]), dim=0).clone()
					reward_tensor = torch.cat((reward_tensor, exp[2]), dim=0).clone()
					new_S_tensor = torch.cat((new_S_tensor, exp[3]), dim=0).clone()
				action_arr[i, :] += exp[1]
			
			# Calculate Q(S, A)
			extract_tensor = np.zeros((3, 1))
			extract_tensor[0, 0] = 1
			extract_tensor = torch.tensor(extract_tensor, device=device, dtype=dtype)
			action_batch = torch.tensor(action_arr, device=device, dtype=torch.long)
			# print("All action value:\n", policy_net(S_tensor))
			# print(policy_net(S_tensor).gather(1, action_batch))
			# print("extract_tensor: \n", extract_tensor)
			action_value_batch = policy_net(S_tensor).gather(1, action_batch).mm(extract_tensor)
			# print("action_value_batch: \n", action_value_batch)
			
			# Calculate max Q(s', a)
			next_action_value_batch = target_net(new_S_tensor).max(1)[0].detach()
			# print("next_action_value_batch:\n", next_action_value_batch.unsqueeze(1))
			q_learning_target = reward_tensor + gamma * next_action_value_batch.unsqueeze(1)
			# print("q_learning_target:\n", q_learning_target)
			# loss = (q_learning_target - action_value_batch).pow(2).sum()
			loss = F.smooth_l1_loss(action_value_batch, q_learning_target)
			# print("loss: \n", loss)
			# TODO: store loss
			if step % 500 == 0:
				print("Step %d\tloss = %f" % (step, loss.item()))

			optimizer.zero_grad()
			# backpropagate loss using autograd
			loss.backward()
			
			optimizer.step()

		# Move on
		state = new_state.clone()
		step += 1
	
	print("Episode %d finished after %d step" % (i_episode, step))
	print("----------------------------------------")
	step_per_episode.append(step)

	if i_episode % update_target_net == 0:
		target_net.load_state_dict(policy_net.state_dict())


plt.figure(1)
plt.plot(step_per_episode)
plt.show()

for k in act_dict.keys():
	if k % 10 == 0:
		plt.figure(k)
		plt.plot(act_dict[k], 'o')
		plt.show()

def plotValueFunc():
	x_vect = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=50)
	xdot_vect = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=50)
	x, xdot = np.meshgrid(x_vect, xdot_vect)
	state_value = np.zeros(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			state = np.array([x[i, j], xdot[i, j]])
			state = torch.tensor(state.reshape((1, 2)), device=device, dtype=dtype)  
			action_value_tensor = target_net(state)
			action_value_list = action_value_tensor.tolist()[0]
			act, p = epsilonGreedy(action_value_list, epsilon, outputprob=True)
			for act in range(env.action_space.n):	
				state_value[i, j] += p[act] * action_value_list[act]
	# plot Value function
	fig = plt.figure(2)
	ax = fig.gca(projection='3d')
	surf = ax.plot_surface(x, xdot, -state_value, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax.set_xlabel('Position')
	ax.set_ylabel('Velocity')
	ax.set_zlabel('Value')
	ax.set_title("Mountain \"Cost To Go\" Function")
	plt.show()

plotValueFunc()
