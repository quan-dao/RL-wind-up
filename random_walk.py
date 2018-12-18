import numpy as np
import matplotlib.pyplot as plt


def monteCarlo(alpha, gamma=0.99, epsilon=1e-3):
	num_states = 7
	value_func = np.zeros(num_states) + 0.5  # initialize V
	value_func[0] = 0
	value_func[-1] = 0
	converge = False
	ep = 0
	ep_max = 1000


	def playOneEp(gamma):
		num_states = 7
		traj = [3]
		i = 3  # initial position
		visitted = np.zeros(num_states) - 1  # list of first time-step at which each cell is visitted 
		visitted[i] = 0  # 
		ret = np.zeros(num_states)  # store reward for each state
		t = 0  # time step
		reach_terminal = False
		while not reach_terminal:
			dice = np.random.choice([0, 1], size=(1, ), p=[0.5, 0.5])
			# move to new position
			i += np.power(-1, int(dice) + 1)
			
			# recieve reward
			if i == 6:
				r = 1
			else:
				r = 0
			# add up return
			for k in range(1, 6):
				if visitted[k] > -1:
					ret[k] += np.power(gamma, t - visitted[k]) * r
			
			# increase time step
			t += 1

			# check if this position is already visitted, if not, mark new position as visitted
			if visitted[i] == -1:
				visitted[i] = t

			# check for terminal state
			if i == 0 or i == 6:
				reach_terminal = True

		# 	# print return
		# 	print("Step %d :\t" % (t - 1), ret)
			traj.append(i)

		# print("Trajectory: ", traj)
		return ret, traj


	while not converge and ep < ep_max:
		last_value_func = np.zeros(num_states) + value_func
		# print("initial last_value_func", last_value_func)
		# execute 1 episode
		ret, traj = playOneEp(gamma)
		value_func += alpha * (ret - value_func)
		# Tight value function of 2 terminal states
		value_func[0] = 0
		value_func[-1] = 0
		# check for convergence
		if np.sum((value_func > 0).astype(int)) == 6 :  # prevent trivial value func
			j = 1
			while j < 6:
				if abs(value_func[j] - last_value_func[j]) > epsilon:
					break
				else:
					j += 1
			if j == 6:
				# print("convergence !!!!")
				# print(last_value_func)
				# print(value_func)
				converge = True
		# increase ep
		ep += 1 

		print("Episode %d:" % ep)
		# print(ret)
		print(value_func)
		print("=====================================")
	if converge:
		print("Stop because of convergence")
	else:
		print("Reaching max episode")

	return value_func


def temporalDifferent0(alpha, gamma=0.99, epsilon=1e-3):
	num_state = 7
	value_func = np.zeros(num_state) + 0.5
	value_func[0] = 0
	value_func[-1] = 0
	convert = False
	it = 0
	it_max = 1000
	while not convert and it < it_max:
		last_value_func =  0 + value_func
		for i in range(num_state):
			# choose action randomly
			dice = np.random.choice([0, 1], size=(1, ), p=[0.5, 0.5])
			# move to new position
			if i > 0 and i < 6:
				j = i + np.power(-1, int(dice) + 1)
				# get reward
				if j == 6:
					r = 1
				else:
					r = 0
				# update value function
				value_func[i] += alpha * (r + gamma * last_value_func[j] - last_value_func[i])
				# print("value_func = ", value_func)
		# check for convergence
		if np.sum((value_func > 0).astype(int)) == 6: 
			delta_val = np.abs(value_func - last_value_func)
			convert = np.sum((delta_val > epsilon).astype(int)) == 0
		# increase it
		it += 1
		print("Iteration ", it)
		print("last value function")
		print(last_value_func)
		print("value function")
		print(value_func)
		print("============================")

	# if it < it_max:
	# 	print("Stop because of convergence.")
	# else:
	# 	print("Stop because of reaching maximum iteration")

	return value_func



if __name__ == '__main__':
	mc_val_func = monteCarlo(0.04)
	td_val_func = temporalDifferent0(0.01)

	plt.plot(mc_val_func[1: -1], label='MC')
	plt.plot(td_val_func[1: -1], label='TD')
	plt.legend()
	plt.show()
