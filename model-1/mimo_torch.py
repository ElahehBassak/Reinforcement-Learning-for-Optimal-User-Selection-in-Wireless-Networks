# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:22:25 2024

@author: Hossein
"""

#########################################
#              Import Library           #
#########################################
import numpy as np
import torch
import time
import math
import os
import matplotlib
import matplotlib.pyplot as plt

#########################################
#            Global Parameters          #
#########################################

# Waveform params
N_OFDM_SYMS             = 24           # Number of OFDM symbols
# MOD_ORDER               = 16           # Modulation order (2/4/16/64 = BSPK/QPSK/16-QAM/64-QAM)
TX_SCALE                = 1.0          # Scale for Tdata waveform ([0:1])

# OFDM params
SC_IND_PILOTS           = torch.tensor([7, 21, 43, 57])                           # Pilot subcarrier indices
#print(SC_IND_PILOTS)
SC_IND_DATA             = torch.from_numpy(np.r_[1:7,8:21,22:27,38:43,44:57,58:64] ) # Data subcarrier indices
#print(SC_IND_DATA)
N_SC                    = 64                                     # Number of subcarriers
# CP_LEN                  = 16                                    # Cyclic prefidata length
N_DATA_SYMS             = N_OFDM_SYMS * len(SC_IND_DATA)     # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)

SAMP_FREQ               = 20e6

# Massive-MIMO params
# N_UE                    = 4
N_BS_ANT                = 64               # N_BS_ANT >> N_UE
# N_UPLINK_SYMBOLS        = N_OFDM_SYMS
N_0                     = 1e-2
H_var                   = 0.1


# LTS for CFO and channel estimation
lts_f = torch.tensor([0, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1])
# lts_t = np.fft.ifft(lts_f, 64)
#print(lts_t)



#########################################
#      Modulation and Demodulation      #
#########################################

modvec_bpsk   =  (1/torch.sqrt(torch.tensor(2)))  * torch.tensor([-1, 1]) # and QPSK
modvec_16qam  =  (1/torch.sqrt(torch.tensor(10))) * torch.tensor([-3, -1, +3, +1])
modvec_64qam  =  (1/torch.sqrt(torch.tensor(43))) * torch.tensor([-7, -5, -1, -3, +7, +5, +1, +3])


def modulation (mod_order,data):
    return torch.complex(modvec_16qam[data>>2],modvec_16qam[torch.remainder(data,4)])
    '''
    if (mod_order == 2): #BPSK
        return torch.complex(modvec_bpsk[data],0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return torch.complex(modvec_bpsk[data>>1],modvec_bpsk[torch.remainder(data,2)])
    elif (mod_order == 16): #16-QAM
        return torch.complex(modvec_16qam[data>>2],modvec_16qam[torch.remainder(data,4)])
    elif (mod_order == 64): #64-QAM
        return torch.complex(modvec_64qam[data>>3],modvec_64qam[torch.remainder(data,8)])
    '''

def demodulation (mod_order, data):

    if (mod_order == 2): #BPSK
        return float(torch.real(data)>0) # data = 0/1
    elif (mod_order == 4): #QPSK
        return float(2*(torch.real(data)>0) + 1*(torch.imag(data)>0))
    elif (mod_order == 16): #16-QAM
        return float((8*(torch.real(data)>0)) + (4*(abs(torch.real(data))<0.6325)) + (2*(torch.imag(data)>0)) + (1*(abs(torch.imag(data))<0.6325)))
    elif (mod_order == 64): #64-QAM
        return float((32*(torch.real(data)>0)) + (16*(abs(torch.real(data))<0.6172)) + (8*((abs(torch.real(data))<(0.9258))and((abs(torch.real(data))>(0.3086))))) + (4*(torch.imag(data)>0)) + (2*(abs(torch.imag(data))<0.6172)) + (1*((abs(torch.imag(data))<(0.9258))and((abs(torch.imag(data))>(0.3086))))))

## H:(N_BS,N_UE), N_UE scalar, MOD_ORDER:(N_UE,)
def data_process (H, N_UE, MOD_ORDER):
    
    pilot_in_mat = torch.zeros((N_UE, N_SC, N_UE));
    for i in range (0, N_UE):
        pilot_in_mat [i, :, i] = lts_f;
    '''
    lts_f_mat = torch.zeros((N_BS_ANT, N_SC, N_UE));
    for i in range (0, N_UE):
        lts_f_mat[:, :, i] = torch.tile(lts_f, (N_BS_ANT, 1))
    '''
    lts_f_mat= torch.tile(lts_f.reshape(-1,1), (N_BS_ANT, 1 , N_UE))

    ## Uplink
    '''
    # Generate a payload of random integers
    tx_ul_data = torch.zeros((N_UE, N_DATA_SYMS),dtype=torch.long)
    for n_ue in range (0,N_UE):
        tx_ul_data[n_ue,:] = torch.randint(low = 0, high = int(MOD_ORDER[n_ue]), size=(1, N_DATA_SYMS))
    '''
    tx_ul_data = torch.randint(low = 0, high = int(MOD_ORDER[0]), size=(N_UE, N_DATA_SYMS)).to(torch.long)

    # Map the data values on to complex symbols
    #tx_ul_syms = torch.zeros((N_UE, N_DATA_SYMS),dtype=torch.cdouble)
    vec_mod = torch.vmap(modulation)
    '''
    for n_ue in range (0,N_UE):
        #tx_ul_syms[n_ue,:] = vec_mod(MOD_ORDER[n_ue].reshape(1,-1), tx_ul_data[n_ue,:].reshape(1,-1))
        tx_ul_syms[n_ue,:] = modulation(MOD_ORDER[n_ue], tx_ul_data[n_ue,:])
    '''
    tx_ul_syms = vec_mod(MOD_ORDER.reshape(-1,1), tx_ul_data).to(torch.cdouble)
    

    # Reshape the symbol vector to a matrix with one column per OFDM symbol
    tx_ul_syms_mat = torch.reshape(tx_ul_syms, (N_UE, len(SC_IND_DATA), N_OFDM_SYMS))

    # Define the pilot tone values as BPSK symbols
    pt_pilots = torch.t(torch.tensor([[1, 1, -1, 1]]))

    # Repeat the pilots across all OFDM symbols
    '''
    pt_pilots_mat = torch.zeros((N_UE, 4, N_OFDM_SYMS),dtype=torch.cdouble)

    for i in range (0,N_UE):
        pt_pilots_mat[i,:,:] = torch.tile(pt_pilots, (1, N_OFDM_SYMS))
    '''
    pt_pilots_mat = torch.tile(pt_pilots, (N_UE, 1, N_OFDM_SYMS)).to(torch.cdouble)

    # Construct the IFFT input matrix
    data_in_mat = torch.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype=torch.cdouble)

    # Insert the data and pilot values; other subcarriers will remain at 0
    data_in_mat[:, SC_IND_DATA, :] = tx_ul_syms_mat
    data_in_mat[:, SC_IND_PILOTS, :] = pt_pilots_mat
    ################ SUMMARY SO FAR ############################
    # Data is ready for all subcarriers and symbols and users, including pilots and payloads
    # 4 pilots in 64 subcarriers (60 payload), BPSK for pilots and MOD_ORDER for payload
    ############################################################

    #Adding pilot_in_mat as the first symbol of data ! (time = 0)
    tx_mat_f = torch.concatenate((pilot_in_mat, data_in_mat),axis=2)

    # Reshape to a vector
    tx_payload_vec = torch.reshape(tx_mat_f, (N_UE, -1))


    # UL noise matrix
    Z_mat = torch.sqrt(torch.tensor(N_0/2)) * ( torch.rand((N_BS_ANT,tx_payload_vec.shape[1])) + 1j*torch.rand((N_BS_ANT,tx_payload_vec.shape[1])))
    
    rx_payload_vec = torch.matmul(H, tx_payload_vec) + Z_mat
    ############SUMMARY###################
    #Generating receiving signal for all subcarriers and time slots, 
    #Assuming all subcarriers and time slots experiencing same channel condition H
    #####################################
    rx_mat_f = torch.reshape(rx_payload_vec, (N_BS_ANT, N_SC, N_UE + N_OFDM_SYMS))
    

    csi_mat = torch.multiply(rx_mat_f[:, :, 0:N_UE], lts_f_mat)

    fft_out_mat = rx_mat_f[:, :, N_UE:]
    

    # precoding_mat = np.zeros((N_BS_ANT, N_SC, N_UE),dtype='complex')
    demult_mat = torch.zeros((N_UE, N_SC, N_OFDM_SYMS),dtype=torch.cdouble)
    sc_csi_mat = torch.zeros((N_BS_ANT, N_UE),dtype=torch.cdouble)

    #################SUMMARY####################
    #estimating and equalizing channel
    ############################################

    for j in range (0,N_SC):
        sc_csi_mat = csi_mat[:, j, :]
        zf_mat = torch.linalg.pinv(sc_csi_mat)   # ZF
        demult_mat[:, j, :] = torch.matmul(zf_mat, torch.squeeze(fft_out_mat[:, j, :]))


    # # Apply the pilot phase correction per symbol
    # demult_pc_mat = np.multiply(demult_mat, pilot_phase_corr)
    payload_syms_mat = demult_mat[:, SC_IND_DATA, :]
    payload_syms_mat = torch.reshape(payload_syms_mat, (N_UE, -1))
    
    ################SUMMARY###########################
    #payload after equalization (ZF) 
    #################################################


    
    #####################SUMMARY########################
    #calculating EVM given received symbols and original symbols (imagine constellation)
    ####################################################
    tx_ul_syms_vecs = torch.reshape(tx_ul_syms_mat, (N_UE, -1))
    ul_evm_mat = torch.square(torch.abs(payload_syms_mat - tx_ul_syms_vecs))
    ul_aevms = torch.mean(ul_evm_mat, 1)
    ul_snrs = 10*torch.log10(1 / ul_aevms)
    ########################SUMMARY####################
    #calculating EVM and accordingly SNR
    ###################################################
    
    ul_snr_discount = torch.clone(ul_snrs)
    '''
    ul_snr_discount = torch.zeros(N_UE)
    for n_ue in range (0,N_UE):
        if (MOD_ORDER[n_ue] == 4):
            ul_snr_discount[n_ue] = ul_snrs[n_ue] + 10*torch.log10(torch.tensor(4))
        elif (MOD_ORDER[n_ue] == 16):
            ul_snr_discount[n_ue] = ul_snrs[n_ue]
        elif (MOD_ORDER[n_ue] == 64):
            ul_snr_discount[n_ue] = ul_snrs[n_ue] - 10*torch.log10(torch.tensor(4))
        ul_snrs[n_ue] = ul_snrs[n_ue] if ul_snrs[n_ue] > 0 else 0
    '''
    ## Spectrual Efficiency
    ## SUM(min(log2(1+SNR),MOD_ORDER))
    ul_se = torch.log2(1+10**(ul_snrs/10)) * torch.log2(MOD_ORDER)
    '''
    ul_se = torch.zeros(N_UE)
    for n_ue in range (0,N_UE):

        ul_se[n_ue] = torch.log2(1+10**(ul_snrs[n_ue]/10)) * torch.log2(MOD_ORDER[n_ue])
    '''
    ul_se_total = torch.sum(ul_se)

    return ul_se_total.numpy(), torch.min(ul_snr_discount).numpy(),ul_se.numpy()