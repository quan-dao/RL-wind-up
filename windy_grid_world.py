

class gridWorld(object):
	"""docstring for girdWorld"""
	def __init__(self):
		super(girdWorld, self).__init__()
		self.start = 0
		self.goal = 0
		self.row = 7
		self.col = 10 
		self.x_max = self.col - 1
		self.y_max = self.row - 1
		# Declare windy column
		self.wind_1 = [3, 4, 5, 8]
		self.wind_2 = [6, 7]

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
			raise("Invalid action. Action must be in (N, E, S, W)")
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
		