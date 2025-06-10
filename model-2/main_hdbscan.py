import argparse
import datetime
import time
import numpy as np
from itertools import combinations
from itertools import product
import torch
import h5py
from sac import SAC
from torch.utils.tensorboard import SummaryWriter

#1-replacing mimo_sim_ul with mimo_torch - optimized implementation of mimo_sim_ul to speed up training 
#from mimo_sim_ul import *
from mimo_torch import *  

#2-replace usr_group with usr_group_hdbscan
#from usr_group import *
from usr_group_hdbscan import *

# from comb import *
from replay_memory import ReplayMemory
#from channel_eigenvalue import *

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="QuaDriGa_SAC_KNN",
                    help='QuaDriGa_SAC_KNN')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.00003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha_lr', type=float, default=0.00003, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--max_episode_steps', type=int, default=500, metavar='N',
                    help='maximum number of steps (TTI) (default: 1000)')
parser.add_argument('--max_episode', type=int, default=1000, metavar='N',
                    help='maximum number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--save_per_epochs', type=int, default=15, metavar='N',
                    help='save_per_epochs')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', default = 0,
                    help='run on CUDA (default: False)')
#support for mac
parser.add_argument('--mps', default = 1,
                    help='run on mps (default: False)')
parser.add_argument('--gpu_nums', type=int, default=2, help='#GPUs to use (default: 1)')
args = parser.parse_args()

# Environment

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)



### Import data from hdf5 file #############################

# Import se_max and H [TTI, BS, UE]
#3- Import full data for all 52 PRBs for High and Low mobility scenario
H_file_low = h5py.File('/Users/sara/Documents/ITU-Challenge/Updated-Code-Torch/2024-ITU-Challenge-UofTW-/Dataset/Processed_datasets/Low_Mob_pre_process_full.hdf5','r')
H_file_high = h5py.File('/Users/sara/Documents/ITU-Challenge/Updated-Code-Torch/2024-ITU-Challenge-UofTW-/Dataset/Processed_datasets/High_Mob_pre_process_full.hdf5','r')
H_r_low = np.array(H_file_low.get('H_r'))
H_i_low = np.array(H_file_low.get('H_i'))
H_r_high = np.array(H_file_high.get('H_r'))
H_i_high = np.array(H_file_high.get('H_i'))

H_low = np.array(H_r_low + 1j*H_i_low)
H_high = np.array(H_r_high + 1j*H_i_high)

print("H shape is:", H_high.shape)

se_max_ur_low = H_file_low.get('se_max')
se_max_ur_low = np.array(se_max_ur_low)

se_max_ur_high = H_file_high.get('se_max')
se_max_ur_high = np.array(se_max_ur_high)

print("se_max_ur shape is:", se_max_ur_high.shape)


normal_ur_low = np.array(H_file_low.get('normal'))
normal_ur_high = np.array(H_file_high.get('normal'))

print("normal_ur shape is:", normal_ur_high.shape)


#############################################################
num_ue = 64
num_bs = 64

#4- Adjust num_actions & max_actions
max_actions = 16#256 
num_states = 192
num_actions = 16#8
random = 1
epsilon = 10000

# Agent
agent = SAC(num_states, num_actions, max_actions, args)

#Tensorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

### User space
user_set = np.r_[0:num_ue]
mod_set = [16]

### Reward Function Parameters
a = 1
b = 1
c = 0
reward_scale = 0.1

### Record vector defination
reward_record = np.zeros((args.max_episode,))
history_record = np.zeros((10,args.max_episode_steps,num_ue))
max_history_record = np.zeros((args.max_episode_steps,num_ue))
ue_history = 0.01*np.ones((num_ue,)) 
idx = 0
ue_select = []

for i_episode in range (args.max_episode): 
   
   #5- randomly select H (scenario (high, low) mobility & randomly select one subcarrier)
    #steps: 
       #1-draw a binary random variable (H_index1) from a bernoulli distribution with P=0.8 (P(H_index1=1)=0.8, & P(H_index1=0)=0.2). we select H from high mobility with P=0.8 & H from low mobility with P=0.2.
       #2- draw an integer random variable (H_index2) between 0 & 40 from a discrete uniform distribution.  H_index2 represents the PRB index, that is at each itteration we select a PRB randomly 
    H = None
    H_index1 = np.random.choice(2, p=[0.2,0.8])
    H_index2 = np.random.randint(40)
    if H_index1 == 0:
        H = H_low[:,:,:,H_index2]
        se_max_ur = se_max_ur_low[:,:,H_index2]
        normal_ur = normal_ur_low[:,H_index2]
    else:
        H = H_high[:,:,:,H_index2]
        se_max_ur = se_max_ur_high[:,:,H_index2]
        normal_ur = normal_ur_high[:,H_index2]
    #print(H.shape)

    episode_reward = 0
    episode_steps = 0
    done = False
    ue_ep_history = np.zeros((args.max_episode_steps,num_ue))
    ue_history = 0.01*np.ones((num_ue,))
    
    #6- use HDBSCAN for user grouping
    #group_idx = usr_group(np.squeeze(H[0,:,:]))
    group_idx = cluster_with_hdbscan(H[0,:,:])
    
    #7- use normalized |H| instead of se_max, and normalized ue_history
    H_mag=np.linalg.norm(H[0,:,:],axis=0) 
    H_mag_norm=(H_mag-H_mag.mean())/(H_mag.std()+1e-8)
    #se_max=se_max_ur[0,:]
    #se_max_norm = (se_max-se_max.mean())/(se_max.std()+1e-8)
    ue_history_norm = (ue_history-ue_history.mean())/(ue_history.std()+1e-8)
    
    #8- use normalized state values to speed up convergence and enhance performance (increase reward)
    state = np.concatenate((H_mag_norm.reshape(1,-1),np.reshape(ue_history_norm,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
    #state = np.concatenate((np.reshape(state_se_norm,(1,num_ue)),np.reshape(state_ue_history,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
    # state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
    # print('state shape is:',state.shape)
    
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        start_time = time.time()
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action, final_action = agent.random_action()
            # print('Random action',final_action)
        else:
            action, final_action = agent.select_action(state)
            # print('Actor action',final_action)
        # print('final action is: ', final_action)
        random -= 1/epsilon

        action_bit_array = np.binary_repr((final_action),width=num_ue) ####@Removed -1
        action_bit_array = np.array(list(action_bit_array), dtype=int)
        # print('bit array is:',action_bit_array)

        for i in range (0,num_ue):
            if action_bit_array[i] == 1:
                idx += 1
                ue_select.append(i)

        ue_select = np.array(ue_select)
        # print("UE selection:", ue_select)
        all_user_idx=np.array(range(64))

        # ue_select,idx = sel_ue()
        mod_select = np.ones((idx,)) * 16 # 16-QAM
        # print(H[episode_steps,:,ue_select])
        
        #9: use optimized implementation of data process using mimo_torch to speed up training
        ur_se_total, ur_min_snr, ur_se = data_process(torch.from_numpy(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1))).to(torch.cdouble), torch.tensor(idx), torch.from_numpy(mod_select))
        #ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
        
        ur_se_total = ur_se_total/normal_ur[episode_steps]
        # print("Normal_ur[episode_steps]:",normal_ur[episode_steps])

        # print('se_total, min_snr are',ur_se_total, ur_min_snr)

        for i in range(0,idx):
            ue_history[ue_select[i]] += ur_se[i]
            # print('ur_se is',ur_se[i])

        ue_ep_history[episode_steps,:] = ue_history

        if (i_episode >= (args.max_episode-10)):
            history_record[i_episode-(args.max_episode-10),episode_steps,:] = ue_history
       
        jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
        # print('jfi is', jfi)
        
        #10: adjust next state parameters- use HDBSCAN, and normalize |H| & ue_history
        group_idx_next = cluster_with_hdbscan(np.squeeze(H[(episode_steps+1),:,:]))
        #group_idx_next = usr_group(np.squeeze(H[(episode_steps+1),:,:]))
        H_mag_next=np.linalg.norm(H[(episode_steps+1),:,:],axis=0) 
        H_mag_norm_next=(H_mag_next-H_mag_next.mean())/(H_mag_next.std()+1e-8)
        
        #next_se_max = se_max_ur[(episode_steps+1),:]
        #next_se_max_norm = (next_se_max-next_se_max.mean())/(next_se_max.std()+1e-8)
        
        ue_history_norm = (ue_history-ue_history.mean())/(ue_history.std()+1e-8)
        
        next_state = np.concatenate((H_mag_norm_next.reshape(1,-1),np.reshape(ue_history_norm,(1,num_ue)),np.reshape(group_idx_next,(1,-1))),axis = 1)
        #next_state = np.concatenate((np.reshape(next_state_se_norm,(1,num_ue)),np.reshape(state_ue_history,(1,num_ue)),np.reshape(group_idx_next,(1,-1))),axis = 1)
        
        # print("NExt state:", next_state)
        reward  = (a*ur_se_total + b*jfi)*reward_scale
        # print("reward terms are:",a*ur_se_total,b*jfi)
        
        done = False
        # print('reward is:', reward)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            
        cal_time = time.time()

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
                
        
        # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        idx = 0
        ue_select = []
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        print('episode reward is:', episode_reward,'\n')

        mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        # end_time = time.time()
        # print('Training time is:', (end_time-cal_time+inf_time-start_time))
    # if total_numsteps > args.num_steps:
    #     break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    reward_record[i_episode] = episode_reward
    if episode_reward == np.max(reward_record):
        max_history_record = ue_ep_history

    if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
        agent.save_checkpoint('sac_checkpoint_UofTW_Model_2_')


with h5py.File("./sac_checkpoint_UofTW_Model_2_.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    data_file.create_dataset("max_history", data=max_history_record)
print('Training is finished\n')




