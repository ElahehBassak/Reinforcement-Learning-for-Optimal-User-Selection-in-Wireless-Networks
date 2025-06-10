# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:37:07 2024

@author: Hossein
"""
from main import args
from sac import SAC
import numpy as np
from mimo_torch import *
import argparse
import datetime
import time
import torch
import h5py



total_begin = time.time()
num_ue = 64
num_bs = 64

max_actions = 4 
num_states = 2080

num_actions = 32

agent = SAC(num_states, num_actions, max_actions, args)

agent.load_checkpoint("./checkpoints/sac_checkpoint_idea_1_4_")
# Training Loop
#total_numsteps = 0

### User space
H_file_low = h5py.File('./Dataset/Low_Mob_500_52_64_64.hdf5','r')
H_file_high = h5py.File('./Dataset/High_Mob_500_52_64_64.hdf5','r')  ## locate your raw datasets here

H_r_low = np.array(H_file_low.get('H_r'))
H_i_low = np.array(H_file_low.get('H_i'))
H_r_high = np.array(H_file_high.get('H_r'))
H_i_high = np.array(H_file_high.get('H_i'))

H_low = np.array(H_r_low + 1j*H_i_low)
H_high = np.array(H_r_high + 1j*H_i_high)
H_low = np.transpose(H_low,(2,1,0,3))
H_high = np.transpose(H_high,(3,1,0,2))
print("H LOW DIM IS", H_low.shape)
print("H HIGH DIM IS", H_high.shape)

##SPECIFY H HERE


H = H_high[:,:,:,50]


##########
### Reward Function Parameters

### Record vector defination

ue_history = 0.01*np.ones((num_ue,)) 
idx = 0
ue_select = []

corr_indices = np.triu_indices(num_ue, k=1)


episode_rate = 0
episode_steps = 0
total_action_time = 0
total_policy_time = 0
total_discretization_time = 0
total_grouping_time = 0
total_env_time = 0
done = False
ue_history = 0.01*np.ones((num_ue,))


t2 = time.time()

#Finding Correlation

H_temp = H[0,:,:]
norm = np.linalg.norm(H_temp, axis = 0)
multip = np.abs(np.matmul(H_temp.conj().T,H_temp))
corr = multip/np.matmul(norm.reshape(-1,1),norm.reshape(1,-1))
## raveling unique corr indices
corr_state = corr[corr_indices]


############

state = np.concatenate((norm.reshape(1,-1),corr_state.reshape(1,-1)),axis = 1)
# print('state shape is:',state.shape)
#print('Evaluation processing: ',i_episode,' episode ',episode_steps,' episode_steps')

while not done:
    #print('Evaluation processing: ',i_episode,' episode ',episode_steps,' episode_steps')
    #print(done)
    
    action, final_action, policy_time, discretization_time = agent.select_action(state)  
    total_policy_time += policy_time
    total_discretization_time += discretization_time


    action_bit_array = np.binary_repr(final_action,width=num_ue)
    action_bit_array = np.array(list(action_bit_array), dtype=int)
    
    # print('bit array is:',action_bit_array)
   

    ue_select = np.where(action_bit_array == 1)[0]
    idx = len(ue_select)
    #print(idx)
            
    # print("UE selection:", ue_select)
    #print(group_idx[ue_select])
    # ue_select,idx = sel_ue()
    mod_select = np.ones((idx,)) *16 # 16-QAM
    # print(H[episode_steps,:,ue_select])
    t1 = time.time()
    total_action_time += (t1-t2) 
    ur_se_total, ur_min_snr, ur_se = data_process(torch.from_numpy(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1))).to(torch.cdouble), torch.tensor(idx), torch.from_numpy(mod_select))
    #ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)), idx, mod_select)
    t2 = time.time()
    
    # print("Normal_ur[episode_steps]:",normal_ur[episode_steps])

    # print('se_total, min_snr are',ur_se_total, ur_min_snr)

    ue_history[ue_select] += ur_se
        # print('ur_se is',ur_se[i])
    #ue_history_norm = ue_history/ue_history.mean()

   
    jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
    # print('jfi is', jfi)
    

    t3 = time.time()
    grouping_time = t3 - t2
    total_grouping_time += grouping_time
    total_env_time += (t2-t1) 
    #Finding Correlation

    H_temp = H[(episode_steps+1),:,:]
    norm = np.linalg.norm(H_temp, axis = 0)
    multip = np.abs(np.matmul(H_temp.conj().T,H_temp))
    corr = multip/np.matmul(norm.reshape(-1,1),norm.reshape(1,-1))
    ## raveling unique corr indices
    corr_state = corr[corr_indices]


    ############
    next_state = np.concatenate((norm.reshape(1,-1),corr_state.reshape(1,-1)),axis = 1)
    # print("NExt state:", next_state)
    # print("reward terms are:",a*ur_se_total,b*jfi)
    done = False
    # print('reward is:', reward)
    if args.max_episode_steps and episode_steps >= 199:
        done = True
        
    
    # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
    episode_steps += 1
    #total_numsteps += 1
    episode_rate += ur_se_total
    #print(episode_reward/episode_steps)
    idx = 0
    ue_select = []
    #print('episode reward is:', episode_reward,'\n')



    state = next_state
total_end = time.time()  
print('total run time is:', total_end - total_begin,'\n')
print('action time is:', total_action_time/episode_steps,'\n')
print('policy time is:', total_policy_time/episode_steps,'\n')
print('discretization time is:', total_discretization_time/episode_steps,'\n')
print('grouping time is:', total_grouping_time/episode_steps,'\n')
print('Env time is:', total_env_time/episode_steps,'\n')
print('episode rate is:', episode_rate/episode_steps,'\n')
print('episode JFI:', jfi,'\n')