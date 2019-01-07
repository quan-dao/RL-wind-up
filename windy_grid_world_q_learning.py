import numpy as np
import pylab as pl
from windy_grid_world import gridWorld, trajectoryDisp
import time


def gridWorldQLearning(world, _start, _terminal, alpha, gamma=0.9, ep_max=200):
	world.setTerminal(_start, _terminal) 
	# initialize Q(s, a)
	q_dict = {}
	for state in range(world.row * world.col):
		q_dict[state] = {}
		for act in world.actions_list:
			if world.checkTerminal(state):
				q_dict[state][act] = 0
			else:
				q_dict[state][act] = np.random.rand()

	def greedyAct(_q_dict):
		greedy_act = ''
		max_q = -1e10
		for act in world.actions_list:
			if _q_dict[act] > max_q:
				greedy_act = act
				max_q = _q_dict[act]
		return greedy_act

	def epsGreedy(episode, q_dict):
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

	ep_wrt_step = []
	trajectory = []
	for ep in range(1, ep_max + 1):
		print("[Episode %d] Start" % ep)
		s = world.start
		trajectory = []
		while not world.checkTerminal(s):
			# choose act according to behaviour policy
			act = epsGreedy(ep, q_dict[s])
			
			# take act, observe s_prime & reward
			s_prime = world.nextState(s, act)
			reward = world.rewardFunc(s_prime)

			# choose act_prime according to target policy
			act_prime = greedyAct(q_dict[s_prime])
			
			# Update Q(s, a)
			q_dict[s][act] += alpha * (reward + gamma * q_dict[s_prime][act_prime] - q_dict[s][act])

			# store trajectory
			trajectory.append(s)

			# update current state
			s = s_prime
			
			# store the index of this episode for plot
			ep_wrt_step.append(ep)
		trajectory.append(world.goal)
		print("[Episode %d] Finish" % ep)
		
	return trajectory, ep_wrt_step


if __name__ == '__main__':
	_start = (3, 0)
	_goal = (3, 7)
	world = gridWorld()
	start_time = time.time()
	trajectory, ep_wrt_step = gridWorldQLearning(world, _start, _goal, alpha=0.5)
	print("Time elapsed: ", time.time() - start_time)
	trajectoryDisp(world, trajectory)
	pl.figure(1)
	pl.plot(ep_wrt_step)
	pl.xlabel("Number of steps taken")
	pl.ylabel("Number of episodes")
	pl.show()
