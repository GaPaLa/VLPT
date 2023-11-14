# Behavioural cloning conditioned on state-conditioned language




from argparse import ArgumentParser
import pickle
import time
import os
import gym
import minerl
import torch as th
import numpy as np
from lib.data_parallel import BalancedDataParallel
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
figure(figsize=(20, 20), dpi=80)

from agent import PI_HEAD_KWARGS, MineRLAgent
from IDM_data_loader_np256 import DataLoader
from lib.tree_util import tree_map
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np 




















# ------------------ MODEL HYPERPARAMETERS
LM_TIMEOUT_RATE = 3  # results in about 7% silence tokens at NeCubS WPM -- wrong calcualteions, is 20% silence
F_SEQ_LEN = 96

L_SEQ_LEN = F_SEQ_LEN//LM_TIMEOUT_RATE
LM_type = "transfo-xl-wt103"

VPT_MODEL_FILE = 'foundation-model-1x.model'
VLPT_WEIGHTS = 'TRAINING/MIXED/VLPT_2300_.weights'
VPT_WIDTH = 1024
DTYPE = th.bfloat16





# -------------------- TRAINING HYPERPARAMETERS
VPT_LEARNING_RATE = 0.00007 # VPT paper did 0.000181 for finetuning: [we are training to a very different task], [VPT uses linear learning rate decay], [] # to keep the LM intact I dont 
warmup_steps = 400 # warmup should be very short since the transformers are pretrained # PaLI uses 1k warmup steps, obviously dont want to do more
BATCH_SIZE = 4
EPOCHS = 5
N_WORKERS = 31 # Needs to be <= number of videos # Ideally more than batch size to create variation in datasets (otherwise, you will get a bunch of consecutive samples)

VPT_WEIGHT_DECAY = 0.039428 # VPT weigh decay. transfoxl weight decay is 
VPT_MAX_GRAD_NORM = 1.0 # VPT says 5.0, transfoXL says 0.25. We will basically c
LM_MAX_GRAD_NORM = 0.25
LM_WEIGHT_DECAY = 0 # none specficd in transfo xl github or paper

EVAL_BATCH_SIZE=4
num_videos = 172

DATASET = 'DATASET/'

TRAINING_PROGRESS = 'TRAINING/MIXED/training_progress'
max_train_steps = (EPOCHS*num_videos*20*60*15)/(F_SEQ_LEN*BATCH_SIZE)    # num steps = number of frames / number of frames per batch# 3*10 mins per video = 600000 ms -> 4687 chunks of 128 frames. want 1000 hours video = 60,000 minutes = 6,000 videos of 10 minutes each
LOSS_REPORT_RATE = 10
EVALUATION_RATE = 100
#higher tha its peak laerning rate. finetuning a multimdodal LM with the same peak lr seems ok according to PaLI,Flamingo but they also train on other tasks, maybe just keep some minecraft data for langauge training?






















# ------------------------------------- USEFUL UTILITIES
def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def save_hidden_states_VPT(video_ids, hidden_state, saved_hidden_states):
    # Unpack the hidden states
    # Iterate over the batch dimension
    for b in range(BATCH_SIZE):
        video_id = video_ids[b]
        video_hidden_state = []
        for l in range(4):
            (hidden_state_1, (hidden_state_2a, hidden_state_2b)) = hidden_state[l]
            #hidden_state_1[:]=False
            #video_id=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
            
            # Get the hidden state for this video
            video_hidden_state_layer = (hidden_state_1[b].clone(), (hidden_state_2a[b].clone(), hidden_state_2b[b].clone()))
            video_hidden_state.append(video_hidden_state_layer)
        # Save the hidden state for this video
        saved_hidden_states[video_id] = video_hidden_state
    
    return saved_hidden_states

def load_hidden_states_VPT(video_ids, saved_hidden_states):
    assert isinstance(video_ids, list)
    assert(len(video_ids)==BATCH_SIZE)
    B = BATCH_SIZE
    T = 128
    E = VPT_WIDTH
    
    # Initialize the hidden states
    hidden_state_1 = [th.zeros([B, 1, T], dtype=th.bool).to(DEVICE)]*4
    hidden_state_2a = [th.zeros([B, T, E], dtype=DTYPE).to(DEVICE)]*4
    hidden_state_2b = [th.zeros([B, T, E], dtype=DTYPE).to(DEVICE)]*4
    
    # Iterate over the batch dimension
    for b in range(B):
        video_id = video_ids[b]
        #video_id=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
        
        # Check if a hidden state has been saved for this video
        if video_id in saved_hidden_states:

            for l in range(4): # repeat for each layer in VPT
              # Get the saved hidden state for this video
              (video_hidden_state_1, (video_hidden_state_2a, video_hidden_state_2b)) = saved_hidden_states[video_id][l]
              
              # Set the hidden state for this video
              hidden_state_1[l][b] = video_hidden_state_1
              hidden_state_2a[l][b] = video_hidden_state_2a
              hidden_state_2b[l][b] = video_hidden_state_2b
        else:
            print("VPT NEW VIDEO SEEN: ADD FRESH INIT. STATE", video_id)

            for l in range(4): # repeat for each layer in VPT
              # Get a new initial hidden state for this video
              
              
              _, (video_hidden_state_2a, video_hidden_state_2b) = policy.initial_state(1)[l]
              
              # Set the initial hidden state for this video
              #print(video_hidden_state_1.shape)
              is_first_frame_true = th.zeros((1, 128), dtype=th.bool).to(DEVICE)
              is_first_frame_true[:,0]=True
              hidden_state_1[l][b] = is_first_frame_true
              hidden_state_2a[l][b] = video_hidden_state_2a
              hidden_state_2b[l][b] = video_hidden_state_2b

    hidden_state = []
    for i in range(4):
        hidden_state_layer = hidden_state_1[l], (hidden_state_2a[l], hidden_state_2b[l])
        hidden_state.append(hidden_state_layer)

    return hidden_state

def load_hidden_states_LM(video_ids, saved_hidden_states):
    assert isinstance(video_ids, list)
    try:
      T = policy.net.LM.transformer.mem_len
      n_layers = policy.net.LM.transformer.n_layer
    except: 
      return None
    B = BATCH_SIZE
    E = 1024

    out_hidden_state = []
    for i in range(n_layers):
        out_hidden_state.append(th.zeros([T,B,E], dtype=DTYPE).to(DEVICE))

    for b, video in enumerate(video_ids):
        #video=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset

        if video in saved_hidden_states:
            hidden_state = saved_hidden_states[video]
        else:
            hidden_state = policy.net.LM.transformer.init_mems(1)
            #print("LM NEW VIDEO SEEN: ADD FRESH INIT. STATE", video)

        for l in range(n_layers):
            #print('\n',out_hidden_state[l].shape)
            #print(hidden_state[l].shape)
            out_hidden_state[l][:T,b,:E] = hidden_state[l].clone().squeeze(1)


    return out_hidden_state

def save_hidden_states_LM(video_ids, hidden_state, saved_hidden_states):
    try:
      T = policy.net.LM.transformer.mem_len
      n_layers = policy.net.LM.transformer.n_layer
    except:
      return None
    B = BATCH_SIZE
    E = 1024

    for b, video in enumerate(video_ids): #frames:
        #video=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
        out_hidden_state = []
        for layer in hidden_state: # rewrite with the new one
            layer_sample = layer[:T,b,:E].clone().unsqueeze(1)
            out_hidden_state.append(layer_sample) # MAKE SURE WE CLONE - we dont want to mutate states that are are in use 

        saved_hidden_states[video] = out_hidden_state

def load_hidden_states_Xattn(video_ids, saved_hidden_states, SEQ_LEN, E):
    #assert hidden_state.shape == [BATCH_SIZE,F_SEQ_LEN or L_SEQ_LEN, 2048 or 1024]
    
    XATTN_MEMLEN=128
    T = XATTN_MEMLEN + SEQ_LEN
    B = BATCH_SIZE

    out_hidden_state = th.zeros([B,T,E], dtype=DTYPE).to(DEVICE) # keys may have different lengths, so we pad with -10 and mask them in VLPT forward

    for b, video in enumerate(video_ids):
        #video=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset

        if video in saved_hidden_states:
            hidden_state = saved_hidden_states[video]
        else:
            hidden_state = th.zeros([1,T,E], dtype=DTYPE).to(DEVICE) # no past keys
            
            #print("LM NEW VIDEO SEEN: ADD FRESH INIT. STATE", video)

        out_hidden_state[b] = hidden_state.clone().squeeze(0)

    return out_hidden_state

def save_hidden_states_Xattn(video_ids, hidden_state, saved_hidden_states):
    #assert hidden_state.shape == [BATCH_SIZE,F_SEQ_LEN or L_SEQ_LEN, 2048 or 1024]
    
    for b, video in enumerate(video_ids): #frames:
        #video=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
        out_hidden_state = hidden_state[b].clone().unsqueeze(0)

        saved_hidden_states[video] = out_hidden_state
    
























def VPT_evaluate():

    if th.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    s1 = th.cuda.Stream()
    s2 = th.cuda.Stream() # for multithreading. When we need to calculate forward pass for training model and original VPT model for KL-divergence, we can do both concurrently.


    ### ---------------------------- initialise dataset and training
    print('BC: starting data loaders')



    ### ---------------------- initialise BLC agent
    print('BC: LOADING VLPT')
    ### VPT INIT
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(VPT_MODEL_FILE)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs, LM_type=LM_type, LM_TIMEOUT_RATE=LM_TIMEOUT_RATE, L_SEQ_LEN=L_SEQ_LEN, dtype=DTYPE)
    agent.load_weights(VLPT_WEIGHTS) #@ DEBUG EVALUATE
    policy = agent.policy # = th.compile(agent.policy)
    policy.eval()


    ## Data Loader init
    eval_data_loader = DataLoader(
                                    dataset_dir=DATASET+'train/',
                                    n_workers=N_WORKERS,
                                    batch_size=BATCH_SIZE,
                                    F_SEQ_LEN=L_SEQ_LEN*LM_TIMEOUT_RATE,
                                    LM_TIMEOUT_RATE=LM_TIMEOUT_RATE,
                                    LM_SILENCE_TOKEN=2,
                                    n_epochs=1,
                                    start_time=120)
    
    with th.no_grad():
        
        # put VLPT into testing mode
        policy.eval()
        
        # we dont want to disrupt internal states of training during eval so we use fresh ones
        eval_current_video_group_id = 0
        eval_VPT_loss =0
        noise_eval_VPT_loss =0  
        eval_LM_loss =th.tensor(0.).to(DEVICE)
        noised_eval_LM_loss =th.tensor(0.).to(DEVICE)
        
        eval_is_first_frame = th.zeros((EVAL_BATCH_SIZE, F_SEQ_LEN), dtype=th.bool).to(DEVICE)
        num_batch=0
        for batch_i, (eval_video_group_id, eval_subseq_ids, eval_batch_frames, eval_batch_words, eval_batch_actions, _) in enumerate(eval_data_loader):
            num_batch+=1
            if eval_video_group_id != eval_current_video_group_id:
                eval_current_video_group_id = eval_video_group_id

                eval_VPT_state = policy.initial_state(EVAL_BATCH_SIZE)
                eval_LM_state = None
                noise_eval_VPT_state = policy.initial_state(EVAL_BATCH_SIZE)
                noise_eval_LM_state=None
                
                eval_Xattn1_hidden_state = th.zeros([EVAL_BATCH_SIZE,128+F_SEQ_LEN,VPT_WIDTH], dtype=DTYPE).to(DEVICE)
                eval_Xattn2_hidden_state = th.zeros([EVAL_BATCH_SIZE,128+L_SEQ_LEN,1024], dtype=DTYPE).to(DEVICE)
                
                noise_eval_Xattn1_hidden_state = th.zeros([EVAL_BATCH_SIZE,128+F_SEQ_LEN,VPT_WIDTH], dtype=DTYPE).to(DEVICE)
                noise_eval_Xattn2_hidden_state = th.zeros([EVAL_BATCH_SIZE,128+L_SEQ_LEN,1024], dtype=DTYPE).to(DEVICE)

            ### ------------- format input from data loader to agent        
            # format input frames
            eval_batch_frames['img'] = th.from_numpy(eval_batch_frames['img']).to(DTYPE).to(DEVICE)
            # format input/label words
            x_words, y_words = eval_batch_words['input_ids'], eval_batch_words['labels']
            x_words=th.from_numpy(x_words).to(DEVICE)
            y_words=th.from_numpy(y_words).to(DEVICE)
            
            noised_x_words = x_words.clone()
            noised_x_words[:] = 2#th.roll(noised_x_words, 1, 0)
            noised_y_words = y_words.clone()
            noised_y_words = th.roll(noised_y_words, 1, 0)
            moised_frames={'img:',eval_batch_frames['img'].clone()} 

            eval_batch_actions['camera'] = eval_batch_actions['camera'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN,2])
            eval_batch_actions['buttons'] = eval_batch_actions['buttons'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN,20])
            eval_action_labels = agent._IDM_action_to_env(eval_batch_actions)
            #print('\n\n',action_labels)
            eval_action_labels = agent._env_action_to_agent(eval_action_labels, to_torch=True, check_if_null=True)



            ### ----------- FORWARD VLPT - NOISED
            th.cuda.empty_cache()
            th.cuda.synchronize()
            with th.cuda.stream(s1):
              noise_eval_VLPT_pd_action, _, _, noise_eval_VPT_state, noise_eval_LM_state, noised_e_LM_loss, noise_eval_Xattn1_hidden_state, noise_eval_Xattn2_hidden_state = policy.get_output_for_observations(
                ob_words=noised_x_words,
                ob_frames=eval_batch_frames,
                VPT_state=noise_eval_VPT_state,
                LM_state=noise_eval_LM_state,
                LM_labels=noised_y_words,
                first=eval_is_first_frame.clone(),
                Xattn1_state=noise_eval_Xattn1_hidden_state,
                Xattn2_state=noise_eval_Xattn2_hidden_state)

            ### ----------- FORWARD VLPT - CLEAN
            with th.cuda.stream(s2):
              eval_VLPT_pd_action, _, _, eval_VPT_state, eval_LM_state, e_LM_loss, eval_Xattn1_hidden_state, eval_Xattn2_hidden_state = policy.get_output_for_observations(
                ob_words=x_words,
                ob_frames=eval_batch_frames,
                VPT_state=eval_VPT_state,
                LM_state=eval_LM_state,
                LM_labels=y_words,
                first=eval_is_first_frame.clone(),
                Xattn1_state=eval_Xattn1_hidden_state,
                Xattn2_state=eval_Xattn2_hidden_state)
            th.cuda.synchronize()
            th.cuda.empty_cache()


            eval_VLPT_pd_action['buttons'] = eval_VLPT_pd_action['buttons'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN, 1, 1, 8641])
            eval_VLPT_pd_action['camera'] = eval_VLPT_pd_action['camera'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN, 1, 1, 121])

            noise_eval_VLPT_pd_action['buttons'] = noise_eval_VLPT_pd_action['buttons'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN, 1, 1, 8641])
            noise_eval_VLPT_pd_action['camera'] = noise_eval_VLPT_pd_action['camera'].reshape([EVAL_BATCH_SIZE*F_SEQ_LEN, 1, 1, 121])

    
            # calculate loss
            eloss = -policy.get_logprob_of_action(eval_VLPT_pd_action, eval_action_labels).mean().item()
            eval_VPT_loss += eloss 
            noise_eval_VPT_loss += -policy.get_logprob_of_action( noise_eval_VLPT_pd_action, eval_action_labels).mean().item()
            eval_LM_loss += e_LM_loss
            noised_eval_LM_loss += noised_e_LM_loss

            print("BLC_EVAL: batch done!", eval_VPT_loss/num_batch, (eval_LM_loss/num_batch).item() )
            print("                     ", (noise_eval_VPT_loss-eval_VPT_loss)/num_batch,  ((noised_eval_LM_loss-eval_LM_loss)/num_batch).item() )

        
        eval_VPT_loss /= num_batch
        noise_eval_VPT_loss /= num_batch
        eval_LM_loss /= num_batch
        noised_eval_LM_loss /= num_batch

        # return VLPT to training mode ## enable dropout for apporopriate layers
        policy.train()

        return eval_VPT_loss, noise_eval_VPT_loss, eval_LM_loss.cpu().numpy(), noised_eval_LM_loss.cpu().numpy()
        

    # agent estimate 10 video sequence batches of 512 with same tgt_len and mem_len








if __name__ == "__main__":
    VPT_evaluate()