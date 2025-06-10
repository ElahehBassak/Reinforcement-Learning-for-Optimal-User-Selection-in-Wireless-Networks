# Submitted Model 1: Modified SAC Architecture & Code Optimization

# Model Overview

This model modifies the following SAC parameters with respect to the baseline model. 
- SAC Action: number of action Dimensions (num_action) & respectively (max_actions). More details provided in "Action" subsection under the "SAC Agent" Section. 
- SAC state: uses the upper triangle of the correlation matrix & the L2 norm of the users' channel vector. More details provided in "State" subsection under the "SAC Agent" Section. 
- SAC Reward: uses the normalized sum rate, omit JFI fairness criterion. More details provided in "Reward" subsection under the "SAC Agent" Section. 
- Actor and Critic Architectures: uses different architecture for Actor & Critic Networks which is further explained in "Actor Critic Neural NN Architecture" section.
- Training: used both high mobility and low mobility datasets. Also, different subcarriers. More details provided in "Training Procedure" Section.

Also, to expedite the training process (data_process function), we optimized the implementation of mimo_sim_ul using torch (optimized code can be found in mimo_torch.py). 

# SAC Agent

**Action:** The baseline method uses SAC with KNN discretization to improve the precision of the model. The idea of utilizing SAC for the suggested problem is convincing enough for us to use a similar approach for our model with some changes. Instead, of using KNN to improve the action accuracy, we increased the dimension of the action space (num_actions) to achieve this goal. In the baseline KNN discretization, the agent calculates the Q-score for each legitimate neighbor action and selects the one with the highest Q-score. This approach increases the precision of action selection with the significant cost of time efficiency (due to running critic network for each action). With this in mind, to increase the precision of the action selection and improve time efficiency at the same time, we simply increase the number of actions and select the single closest valid action using search_point method.

**State:** To avoid using high dimensional state space, the baseline approach aims to distill the information regarding the channel state matrix by using user_grouping algorithm. Despite the successful attempt to reduce the state space, the amount of information loss in utilizing this method is significant, which deteriorates the performance of agent in selecting optimal actions. To overcome this issue, we use the norm of the users' channel vector alongside the correlation matrix (only upper triangular indices) between all the existing users in which full information about channel state is provided. Moreover, we removed the UE history from the state space as the stochastic nature of the actions in SAC solves the issue of fairness.

**Reward:** Since the stochastic policy of SAC approach already improves the fairness, we omit JFI fairness criterion from the reward function, i.e. reward function has only spectral efficiency component. 

# Actor and Critic NN Architectures

We used the similar architecture for actor network as the baseline  with the difference of increasing the number of neurons of the first intermediate layer (increased to 512 from 256) as our state space is slightly more complex. However, as far as critic network is concerned, we defined separate paths for action and state in our critic network since our state dimension is significantly larger than our action state. We can interpret this approach as performing simple feature extraction on a high-dimensional state vector. The details of architectures can be found on model.py. 

# Training Procedure

Baseline training only happens on 21st subcarrier of high mobility dataset, which possibly makes the model overfit and not generalize well. To find a way around this issue, we perform the training on the first 40 subcarriers of high-mobility and low-mobility datasets (80 subcarriers in total). For each episode, we randomly select high-mobility or low-mobility datasets with probabilities of 0.8 and 0.2, respectively. After which, we employ uniform distribution to select one of the 40 subcarriers of the selected dataset (either low-mobility or high-mobility dataset) randomly. 

# Parameters for 64 single antenna users and 64 BS antennas
We used the following parameters to train our model,

max_episode = 1400

num_states = 64 (norm vector size) + 2016 (upper triangular indices excluding diagonal of correlation matrix) = 2080

num_actions = 32

max_actions = 4 (2 bits per continuous action)

GPU = NVIDIA GeForce RTX 3070

# infer.py

Unlike baseline, our model doesn't require pre-processed data for inference, all it needs is the channel matrix for different time slots of an episode.

# Codes

Some parts of the provided baseline codes seemed slow in our view (namely SAC knn_action and random_action), therefore we replaced those portions with faster implementations. We also optimized the implementation of mimo_sim_ul using torch to expedite the training process (optimized code can be found in mimo_torch.py).
