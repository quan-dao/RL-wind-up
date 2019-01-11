import torch
import numpy as np
import gym


dtype = torch.float
device = torch.device("cuda:0")


env = gym.envs.make("MountainCar-v0")
env._max_episode_steps = 500


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


max_episode = 1
epsilon_start = 0.9  # those are the values for epsilon decay
epsilon_stop = 0.01
epsilon_decay_step = 10
epsilon = epsilon_start
memory = ReplayMemory()
update_frozen_step = 50
gamma = 0.99

batch_size, n_in, n_hidden1, n_hidden2, n_out = 50, 2, 100, 50, 3  # n_in = dim of obs space, n_out = dim of action space
learning_rate = 1e-3

policy_net = torch.nn.Sequential(
	torch.nn.Linear(n_in, n_hidden1),
	torch.nn.Linear(n_hidden1, n_hidden2),
	torch.nn.Linear(n_hidden2, n_out)
)

target_net = torch.nn.Sequential(
	torch.nn.Linear(n_in, n_hidden1),
	torch.nn.Linear(n_hidden1, n_hidden2),
	torch.nn.Linear(n_hidden2, n_out)
)
target_net.load_state_dict(policy_net.state_dict())

policy_net.to(device)
target_net.to(device)
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(policy_net.parameters(), lr = 0.01, momentum=0.9)
# state = env.observation_space.sample()
# state = torch.tensor(state, device=device, dtype=dtype)
# print("state: ", state)
# print("forward: ", policy_net(state))
# action_value_list = policy_net(state).tolist()
# print("action_value_list: ",action_value_list)

step_per_episode = []
for i_episode in range(1,50):
	state = env.reset()
	state = torch.tensor(state.reshape((1, 2)), device=device, dtype=dtype)  # convert state to tensor
	done = False
	step = 0
	# calculate epsilon
	epsilon = return_decayed_value(epsilon_start, epsilon_stop, i_episode, epsilon_decay_step)
	# epsilon = 1./i_episode
	while not done:
		# choose A from S using behaviour policy
		action_value_tensor = policy_net(state)
		# print("action_value_list: ", action_value_tensor.data.tolist())
		action = epsilonGreedy(action_value_tensor.data.tolist()[0], epsilon)
		# print("Episode %d action: " %i_episode, action)

		# take A, observe R and S'
		new_state, reward, done, _ = env.step(action)
		if new_state[0] > 0.:
			reward = 1.
			done = True
			break

		new_state = torch.tensor(new_state.reshape((1, 2)), device=device, dtype=dtype)
		reward = torch.tensor([[reward]], device=device, dtype=dtype)
		# Update memory
		memory.add(state, action, reward, new_state)

		# Experience Replay
		num_exp = min(batch_size, len(memory.data))
		S_tensor = None
		reward_tensor = None
		new_S_tensor = None
		action_arr = np.zeros((num_exp, 3))
		for i, exp in enumerate(memory.sample(num_exp)):
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
		loss = loss_fn(q_learning_target, action_value_batch)
		# print("loss: \n", loss)
		# TODO: store loss
		# if step % 500 == 0:
		# 	print("Step %d\tloss = %f" % (step, loss.item()))

		optimizer.zero_grad()
		# backpropagate loss using autograd
		loss.backward()
		optimizer.step()

		if step % update_frozen_step == 0:
			target_net.load_state_dict(policy_net.state_dict())

		# Move on
		state = new_state.clone()
		step += 1
	print("Episode %d finished after %d step" % (i_episode, step))
	step_per_episode.append(step)



