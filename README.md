## Reinforcement Learning Windup

This repo store my getting started with Reinforcement Learning (RL) code. These code are my implementation of the algorithms presented in the RL course by David Silver (http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

[//]: # (Image References)

[image1]: ./misc/windy_grid_world_sarsa.png
[image2]: ./misc/windy_grid_world_sarsa_traj.png
[image3]: ./misc/windy_grid_world_q_learning.png
[image4]: ./misc/windy_grid_world_q_learning_traj.png
[image5]: ./misc/windy_gird_world.png


### Windy Grid World
This world is described by Fig.1. In this figure, the number associated with each column denotes how many cell the agent will be below up if an agent stands at that column. The agent has four possible actions to North, East, South or West.

![alt text][image5]

*Fig.1 The windy grid world*

#### Solution by SARSA On-policy 
The implementation of SARSA on-polciy control is the file *widny_grid_world.py*. The trajectory found by SARSA after playing 200 episode is displayed in Fig.2.

![alt text][image2]

*Fig.2 The trajectory found by SARSA*

The number of steps and the associated episode index is shown in Fig.3.

![alt text][image1]

*Fig.3 SARSA's learning curve*

#### Solution by Q-Learning (off-policy) 
The implementation of Q-learning is the file *widny_grid_world_q_learning.py*. The trajectory found by this algorithm after playing 200 episode is displayed in Fig.4.

![alt text][image4]

*Fig.4 The trajectory found by Q-learning*

The number of steps and the associated episode index is shown in Fig.5.

![alt text][image3]

*Fig.5 Q-learning's learning curve*

Fig.3 and Fig.5 show that Q-learning takes more step than SARSA to finish 200 episode. However, it can be seen from Fig.2 and Fig.5 that the trajectory found by Q-learning containts less steps than SARSA's.
