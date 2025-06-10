# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:49:31 2024

@author: hosse
"""

import argparse
import datetime
import time
import numpy as np
from itertools import combinations
from itertools import product
import torch
import h5py
from sac import SAC
#from torch.utils.tensorboard import SummaryWriter
from mimo_torch import *
# from comb import *
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt

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
parser.add_argument('--max_episode_steps', type=int, default=499, metavar='N',
                    help='maximum number of steps (TTI) (default: 500)')
parser.add_argument('--max_episode', type=int, default=1400, metavar='N',
                    help='maximum number of episodes (default: 1300)')
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
parser.add_argument('--cuda', default = 1,
                    help='run on CUDA (default: True)')
parser.add_argument('--gpu_nums', type=int, default=1, help='#GPUs to use (default: 1)')
args = parser.parse_args()

# Environment




if __name__ == '__main__':
    
    ### Import data from hdf5 file #############################

    # Import se_max and H [TTI, BS, UE]

    H_file_low = h5py.File('C:/Users/Hossein/OneDrive - University of Toronto/MIMO_Project/SMART-Scheduler-master/ML-Challenge/ML_Challenge_Code/SAC_KNN/Dataset/Low_Mob_pre_process_full.hdf5','r')
    H_file_high = h5py.File('C:/Users/hossein/OneDrive - University of Toronto/MIMO_Project/SMART-Scheduler-master/ML-Challenge/ML_Challenge_Code/SAC_KNN/Dataset/High_Mob_pre_process_full.hdf5','r')
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
    
    max_actions = 4 
    num_states = 2080 

    num_actions = 32
    random = 1
    epsilon = 10000
    
    ### Reward Function Parameters
    #a = 1
    #b = 0.2 
    reward_scale = 0.1


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
    

    ### Record vector defination
    reward_record = np.zeros((args.max_episode,))
    #history_record = np.zeros((10,500,num_ue))
    max_history_record = np.zeros((args.max_episode_steps,num_ue))
    idx = 0
    ue_select = []
    
    
    corr_indices = np.triu_indices(num_ue, k=1)
    
    
    for i_episode in range (args.max_episode): 
        
        #randomly selecting H
        H = None
        H_index1 = np.random.choice(2, p=[0.2,0.8])
        H_index2 = np.random.randint(50)
        if H_index1 == 0:
            H = H_low[:,:,:,H_index2]
            se_max_ur = se_max_ur_low[:,:,H_index2]
            normal_ur = normal_ur_low[:,H_index2]
        elif H_index1 == 1:
            H = H_high[:,:,:,H_index2]
            se_max_ur = se_max_ur_high[:,:,H_index2]
            normal_ur = normal_ur_high[:,H_index2]
        #print(H.shape)
        
        ###############################
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        episode_reward = 0
        episode_steps = 0
        done = False
        ue_ep_history = np.zeros((args.max_episode_steps,num_ue))
        ue_history = 0.01*np.ones((num_ue,))

        
        #Finding Correlation
        
        H_temp = H[0,:,:]
        norm = np.linalg.norm(H_temp, axis = 0)
        multip = np.abs(np.matmul(H_temp.conj().T,H_temp))
        corr = multip/np.matmul(norm.reshape(-1,1),norm.reshape(1,-1))
        ## raveling unique corr indices
        corr_state = corr[corr_indices]
        
        ############
        state = np.concatenate((norm.reshape(1,-1), corr_state.reshape(1,-1)),axis = 1)

        
        # print('state shape is:',state.shape)
        while not done:
            print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
            
            if random > np.random.rand(1):
                # if step <= warmup or (episode < 100):
                action, final_action = agent.random_action()
                # print('Random action',final_action)
            else:
                action, final_action, _, _ = agent.select_action(state)
                # print('Actor action',final_action)
            # print('final action is: ', final_action)
            if final_action == 0:
                continue
            
            random -= 1/epsilon
            
            action_bit_array = np.binary_repr(final_action, width=num_ue)
            action_bit_array = np.array(list(action_bit_array), dtype=int)
            # print('bit array is:',action_bit_array)
        

            ue_select = np.where(action_bit_array == 1)[0]
            idx = len(ue_select)
    

    
            # ue_select,idx = sel_ue()
            mod_select = np.ones((idx,)) *16 # 16-QAM
            # print(H[episode_steps,:,ue_select])
            t1 = time.time()
            action_time = t3 - t2 + t1 - t4
            ur_se_total, ur_min_snr, ur_se = data_process(torch.from_numpy(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1))).to(torch.cdouble), torch.tensor(idx), torch.from_numpy(mod_select)) # first 64 is BS due to transpose
            t2 = time.time()
            ur_se_total = ur_se_total/normal_ur[episode_steps]
            # print("Normal_ur[episode_steps]:",normal_ur[episode_steps])
            
            # print('se_total, min_snr are',ur_se_total, ur_min_snr)

            
            ue_history[ue_select] += ur_se
            
            #ue_history_norm = ue_history - ue_history.mean()
            ue_ep_history[episode_steps,:] = ue_history
            

           
            jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
            # print('jfi is', jfi)
    
            
            #Finding Correlation
            
            H_temp = H[(episode_steps+1),:,:]
            norm = np.linalg.norm(H_temp, axis = 0)
            multip = np.abs(np.matmul(H_temp.conj().T,H_temp))
            corr = multip/np.matmul(norm.reshape(-1,1),norm.reshape(1,-1))
            ## raveling unique corr indices
            corr_state = corr[corr_indices]
            
            ############
            
            next_state = np.concatenate((norm.reshape(1,-1), corr_state.reshape(1,-1)),axis = 1)

            
            # print("NExt state:", next_state)
            reward  = ur_se_total*reward_scale

            done = False
            # print('reward is:', reward)
            if args.max_episode_steps and episode_steps >= args.max_episode_steps-1:
                done = True
                
            
            t3 = time.time()
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
    
                    #writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    #writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    #writer.add_scalar('loss/policy', policy_loss, updates)
                    #writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    #writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
            t4 = time.time()
            # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            idx = 0
            ue_select = []
            # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            print('episode reward is:', episode_reward,'\n')
    
            #mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)
            mask = 1
    
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
    
            state = next_state
            
            print('action time is:', action_time, '\n')
            
            # print('Training time is:', (end_time-cal_time+inf_time-start_time))
        # if total_numsteps > args.num_steps:
        #     break
    
        #writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
        reward_record[i_episode] = episode_reward
        if (i_episode%100) == 0:
            plt.plot(reward_record)
            plt.show()
            
        if episode_reward == np.max(reward_record):
            max_history_record = ue_ep_history
    
        if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
            agent.save_checkpoint('idea_1_4')
    
    with h5py.File("./idea_1_4.hdf5", "w") as data_file:
        data_file.create_dataset("reward", data=reward_record)
        #data_file.create_dataset("history", data=history_record)
        data_file.create_dataset("max_history", data=max_history_record)
    print('Training is finished\n')
    
    
    
