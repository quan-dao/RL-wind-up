import numpy as np


class gridWorld(object):
	"""docstring for gridWorld"""
	def __init__(self):
		super(gridWorld, self).__init__()
		self.start = 0
		self.goal = 0
		self.row = 7
		self.col = 10 
		self.x_max = self.col - 1
		self.y_max = self.row - 1
		# Declare windy column
		self.wind_1 = [3, 4, 5, 8]
		self.wind_2 = [6, 7]
		# action list
		self.actions_list = ['N', 'E', 'S', 'W']

	def cell(self, pos):
		# pos = (y, x)
		return pos[1] + self.col * pos[0]

	def setTerminal(self, _start, _goal):
		# _start & _goal are tuples
		self.start = self.cell(_start)
		self.goal = self.cell(_goal)

	def nextState(self, state, action):
		# state: an integer represents position in grid
		# decode state
		x = state % self.col
		y = (state - x) / self.col
		# interpret action
		del_x = 0
		del_y = 0
		if action == 'E':
			del_x = 1
		elif action == 'W':
			del_x = -1
		elif action == 'N':
			del_y = -1
		elif action == 'S':
			del_y = 1
		else:
			raise("Invalid action. Action must be in ", self.actions_list)
		# move to new position
		new_x = max(0, min(x + del_x, self.x_max))
		new_y = max(0, min(y + del_y, self.y_max))
		
		# let the wind blow to a new state
		if new_x in self.wind_1:
			new_y = max(0, new_y - 1)
		elif new_x in self.wind_2:
			new_y = max(0, new_y - 2)

		# pack new_y & new_x to 1 number then return
		return self.cell((new_y, new_x))

	def checkTerminal(self, state):
		return state == self.goal

	def rewardFunc(self, state_prime):
		if state_prime == self.goal:
			return 0
		else:
			return -1


def gridWorldSarsa(world, _start, _goal, alpha=0.1, gamma=1):
	world.setTerminal(_start, _goal)
	# initialize Q(s, a)
	q_table = {}
	for state in range(world.row * world.col):
		q_table[state] = {}
		for act in world.actions_list:
			q_table[state][act] = 0
	
	# function for greedy action
	def epsGreedy(episode, q_dict):
		
		def greedyAct(_q_dict):
			greedy_act = ''
			max_q = -1e10
			for act in world.actions_list:
				if _q_dict[act] > max_q:
					greedy_act = act
					max_q = _q_dict[act]
			return greedy_act

		eps = 1. / episode
		m = len(world.actions_list)
		greedy_act = greedyAct(q_dict)
		p = []
		for act in world.actions_list:
			if act == greedy_act:
				p.append((eps * 1. / m) + 1 - eps)
			else:
				p.append(eps * 1. / m)
		choice = np.random.choice(world.actions_list, size=1, p=p)
		return choice[0]


	ep = 1
	ep_max = 2
	step = 0
	while ep < ep_max:
		print("Episode ", ep)
		# initialize state
		state = world.cell(_start)
		# choose action from state
		act = epsGreedy(ep, q_table[state])
		while not world.checkTerminal(state):
			state_prime = world.nextState(state, act)
			reward = world.rewardFunc(state_prime)
			act_prime = epsGreedy(ep, q_table[state_prime])
			q_table[state][act] += alpha * (reward + gamma * q_table[state_prime][act_prime] - q_table[state][act])
			state = state_prime
			act = act_prime
			# increase step counter
			step += 1

			# check out 2 state
			print("Step ", step)
			print(q_table[0])
			print("-----------------------------")
			print(q_table[10])
		# increase episode counter
		ep += 1
		print("======================================")

	
	# choice, greedy_act = epsGreedy(1, q_table[10])
	# print("choice: ", choice, "\tgreedy act: ", greedy_act)


if __name__ == '__main__':
	_start = (3, 0)
	_goal = (3, 9)
	world = gridWorld()
	gridWorldSarsa(world, _start, _goal)
