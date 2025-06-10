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
parser.add_argument('--max_episode', type=int, default=1, metavar='N',
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
parser.add_argument('--mps', default = 0,
                    help='run on mps (default: False)')
parser.add_argument('--gpu_nums', type=int, default=2, help='#GPUs to use (default: 1)')
args = parser.parse_args()


### Import data from hdf5 file #############################

# Import se_max and H [TTI, BS, UE]

#use 51th PRB from the pre-process full data set
H_file = h5py.File('/Users/sara/Documents/ITU-Challenge/2024-ITU-Challenge-UofTW-/Dataset/Processed_datasets/Low_Mob_pre_process_full.hdf5','r') 
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))

H_ = np.array(H_r + 1j*H_i)
print("H shape is:", H_.shape)

#use only the 51th PRB
H=H_[:,:,:,50]

se_max_ur_ = H_file.get('se_max')
se_max_ur=se_max_ur_[:,:,50]

se_max_ur = np.array(se_max_ur)
print("se_max_ur shape is:", se_max_ur.shape)

normal_ur_ = np.array(H_file.get('normal'))
normal_ur=normal_ur_[:,50]
print("normal_ur shape is:", normal_ur.shape)

#############################################################
num_ue = 64
num_bs = 64

#3- Adjust the num_actions & max_actions
max_actions = 16#256 
num_states = 192
num_actions = 16#8
random = 0
epsilon = 10000

# Agent
agent = SAC(num_states, num_actions, max_actions, args)

#Tensorboard
#writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))
    
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

ur_se_total_step=np.zeros((args.max_episode, args.max_episode_steps))
jfi_episode=np.zeros((args.max_episode))
run_time_per_step=np.zeros((args.max_episode, args.max_episode_steps))
user_group_time_per_step=np.zeros((args.max_episode, args.max_episode_steps))
action_time_per_step=np.zeros((args.max_episode, args.max_episode_steps))
inference_time_per_step=np.zeros((args.max_episode, args.max_episode_steps))
data_process_time_per_step=np.zeros((args.max_episode, args.max_episode_steps))

j=0

for i_episode in range (args.max_episode): 
    episode_reward = 0
    episode_steps = 0
    done = False
    ue_ep_history = np.zeros((args.max_episode_steps,num_ue))
    ue_history = 0.01*np.ones((num_ue,))

    ckpt_path = "/Users/sara/Documents/ITU-Challenge/Updated-Code-Torch/2024-ITU-Challenge-UofTW-/ML-Challenge/ML_Challenge_Code/SAC_KNN/checkpoints/sac_checkpoint_UofTW_Model_2_" 
    
    agent.load_checkpoint(ckpt_path) #Checkpoints for this model is trained on mac mps
    #----> for loading on cuda, please use the code below<------
    #torch.load(ckpt_path, map_location=torch.device('cuda'))
    
    
    #4- Adjust state input
    
    #use hdbscan for user grouping
    group_idx = cluster_with_hdbscan(H[0,:,:])
    #group_idx = usr_group(np.squeeze(H[0,:,:]))
    
    #use normalized |H| instead of se_max, and normalized ue_history
    H_mag=np.linalg.norm(H[0,:,:],axis=0) 
    H_mag_norm=(H_mag-H_mag.mean())/(H_mag.std()+1e-8)
    #se_max=se_max_ur[0,:]
    #se_max_norm = (se_max-se_max.mean())/(se_max.std()+1e-8)
    ue_history_norm = (ue_history-ue_history.mean())/(ue_history.std()+1e-8)
    
    #use normalized state values to speed up convergence and enhance performance (increase reward)
    state = np.concatenate((H_mag_norm.reshape(1,-1),np.reshape(ue_history_norm,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
    #state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
    
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        step_start_time = time.time()
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action, final_action = agent.random_action()
            # print('Random action',final_action)
            j+=1
        else:
            action, final_action = agent.select_action(state)
            # print('Actor action',final_action)
        # print('final action is: ', final_action)
        random -= 1/epsilon

        action_bit_array = np.binary_repr((final_action),width=num_ue)
        action_bit_array = np.array(list(action_bit_array), dtype=int)
        # print('bit array is:',action_bit_array)
    

        for i in range (0,num_ue):
            if action_bit_array[i] == 1:
                idx += 1
                ue_select.append(i)
              
        action_selection_time=time.time()
        ue_select = np.array(ue_select)
        # print("UE selection:", ue_select)

        # ue_select,idx = sel_ue()
        mod_select = np.ones((idx,)) *16 # 16-QAM
        # print(H[episode_steps,:,ue_select])
        ####end_time = time.time()
        ####run_time_per_step[i_episode,episode_steps-1]=end_time-start_time
        data_process_start_time = time.time()
        
        #use optimized code for data process
        #ur_se_total_, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
        ur_se_total_, ur_min_snr, ur_se = data_process(torch.from_numpy(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1))).to(torch.cdouble), torch.tensor(idx), torch.from_numpy(mod_select))
        
        data_process_end_time = time.time()
        ur_se_total = ur_se_total_/normal_ur[episode_steps]
        # print("Normal_ur[episode_steps]:",normal_ur[episode_steps])
        #save the ur_se_total_ 
        ur_se_total_step[i_episode,episode_steps]=ur_se_total_

        # print('se_total, min_snr are',ur_se_total, ur_min_snr)

        for i in range(0,idx):
            ue_history[ue_select[i]] += ur_se[i]
            # print('ur_se is',ur_se[i])

        ue_ep_history[episode_steps,:] = ue_history

        if (i_episode >= (args.max_episode-10)):
            history_record[i_episode-(args.max_episode-10),episode_steps,:] = ue_history
       
        jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
        print('jfi is', jfi)
      
        user_group_start_time=time.time() 
        
        #5- Adjust state input
        group_idx_next = cluster_with_hdbscan(np.squeeze(H[(episode_steps+1),:,:]))
        #group_idx_next = usr_group(np.squeeze(H[(episode_steps+1),:,:]))
        user_group_end_time=time.time() 
        
        H_mag_next=np.linalg.norm(H[(episode_steps+1),:,:],axis=0) 
        H_mag_norm_next=(H_mag_next-H_mag_next.mean())/(H_mag_next.std()+1e-8)
        ue_history_norm = (ue_history-ue_history.mean())/(ue_history.std()+1e-8)
      
        next_state = np.concatenate((H_mag_norm_next.reshape(1,-1),np.reshape(ue_history_norm,(1,num_ue)),np.reshape(group_idx_next,(1,-1))),axis = 1)
        #next_state = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(group_idx_next,(1,-1))),axis = 1)
        # print("NExt state:", next_state)
        
        reward  = (a*ur_se_total + b*jfi)*reward_scale
        # print("reward terms are:",a*ur_se_total,b*jfi)
        
        done = False
        
        # print('reward is:', reward)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            
        # cal_time = time.time()

        # if len(memory) > args.batch_size:
        #     # Number of updates per step in environment
        #     for i in range(args.updates_per_step):
        #         # Update parameters of all the networks
        #         critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

        #         writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        #         writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        #         writer.add_scalar('loss/policy', policy_loss, updates)
        #         writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        #         writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        #         updates += 1
        
        # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        idx = 0
        ue_select = []
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        print('episode reward is:', episode_reward,'\n')

        # mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)

        # memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        step_end_time = time.time()
        run_time_per_step[i_episode,episode_steps-1]=step_end_time-step_start_time
        user_group_time_per_step[i_episode,episode_steps-1]=user_group_end_time-user_group_start_time
        action_time_per_step[i_episode,episode_steps-1]=action_selection_time-step_start_time
        inference_time_per_step[i_episode,episode_steps-1]=action_time_per_step[i_episode,episode_steps-1]+user_group_time_per_step[i_episode,episode_steps-1]
        data_process_time_per_step[i_episode,episode_steps-1]=data_process_end_time-data_process_start_time
        # print('Training time is:', (end_time-cal_time+inf_time-start_time))
    # if total_numsteps > args.num_steps:
    #     break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # reward_record[i_episode] = episode_reward
    # if episode_reward == np.max(reward_record):
    #     max_history_record = ue_ep_history

    # if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
    #     agent.save_checkpoint('SMART_64_64')

    jfi_episode[i_episode]=jfi 
    ur_se_total_avg_per_step=ur_se_total_step[i_episode].mean()
    avg_run_time_per_step=run_time_per_step[i_episode].mean()
    avg_user_group_time_per_step=user_group_time_per_step[i_episode].mean()
    avg_action_time_per_step=action_time_per_step[i_episode].mean()
    avg_inference_time_per_step=inference_time_per_step[i_episode].mean()
    avg_data_process_time_per_step=data_process_time_per_step[i_episode].mean()
    
    print("Average SE total per TTI: ", ur_se_total_avg_per_step)
    print("JFI per episode: ", jfi_episode[i_episode])
    print("Average Run time per step: ", avg_run_time_per_step)
    print("Average user grouping time per step: ", avg_user_group_time_per_step)
    print("Average Inference time per step(action time + grouping): ", avg_inference_time_per_step)
    print("Average data process time per step: ", avg_data_process_time_per_step)
    #print("number of random actions: ", j)

# with h5py.File("./High_Mob_result.hdf5", "w") as data_file:
#     data_file.create_dataset("reward", data=reward_record)
#     data_file.create_dataset("history", data=history_record)
#     data_file.create_dataset("max_history", data=max_history_record)
print('Inference is finished\n')
