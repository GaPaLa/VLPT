#  train LM to do Language modelling conditioned on the video-processing section of VPT
# also requires training the gated-cross-attention layer between VPT and LM.
# The 'video processing' section on VPT consists of the CNN and the first transformer layer, which are frozen


from argparse import ArgumentParser
import pickle
import time
import os
from typing import List
import gym
import minerl
import torch as th
import numpy as np
from lib.data_parallel import BalancedDataParallel
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
figure(figsize=(20, 20), dpi=80)
th.set_printoptions(precision=10)

from agent import PI_HEAD_KWARGS, MineRLAgent
from IDM_data_loader_np256 import DataLoader
from lib.tree_util import tree_map
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np 










# ------------------ MODEL HYPERPARAMETERS
VPT_MODEL_FILE = 'foundation-model-1x.model' #'VLPT/2x.model'
VPT_WEIGHTS_FILE = 'foundation-model-1x.weights' # 'VLPT/bc-early-game-2x.weights' # 'VLPT/rl-from-early-game-2x.weights' 
DTYPE = th.bfloat16
VPT_WIDTH = 1024

LM_TIMEOUT_RATE = 1  # results in about 7% silence tokens at avg WPM -- wrong calcualteions, is 20% silence
L_SEQ_LEN = 208 # with LM only we can have MUCH longer sequences because we are not holding: any VPT grads, any VPT momentum, any VPT activations/weights past the first transformer block (bc these arent needed to predict the word), any vanvpt weights or activations since we dont need action KL  divergence loss
EVAL_SEQ_LEN = 208 # otherwise we run out of memory
F_SEQ_LEN = L_SEQ_LEN*LM_TIMEOUT_RATE
XATNN_MEMLEN = 256

TRAINING_LOG_FILE = 'training_log'

OUTPUT_WEIGHTS = 'TRAINING/LM_ONLY/VLPT_LM.weights'
# VPT model automatically downloads transfo_xl weigbhts from HuggingFace and uses those for LM. If weights include the LM it should be overwritten though?






# -------------------- TRAINING HYPERPARAMETERS print
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
N_WORKERS = 31 #prime number bc otherwise we get everything batched in the same groups, e.g. video A&B always show up as a batch, C&D always show up as a batch...,  # Needs to be <= number of videos # Ideally more than batch size to create variation in datasets (otherwise, you will get a bunch of consecutive samples)       # -------- ah, just what I'm looking for! # Decrease this (and batch_size) if you run out of memory


num_videos = 172
EPOCHS = 5
DATASET = 'DATASET/'
TRAINING_PROGRESS = 'TRAINING/LM_ONLY/training_progress'
max_train_steps = (EPOCHS*num_videos* 20*60*30 ) / (F_SEQ_LEN*BATCH_SIZE)    # num steps = number of frames / number of frames per batch# 3*10 mins per video = 600000 ms -> 4687 chunks of 128 frames. want 1000 hours video = 60,000 minutes = 6,000 videos of 10 minutes each
warmup_steps = 400 # warmup should be very short since the transformers are pretrained # PaLI uses 1k warmup steps, obviously dont want to do more
LOSS_REPORT_RATE = 10
EVALUATION_RATE = 100
#higher tha its peak laerning rate. finetuning a multimdodal LM with the same peak lr seems ok according to PaLI,Flamingo but they also train on other tasks, maybe just keep some minecraft data for langauge training?

LM_LEARNING_RATE = 0.0001
#tiny, but maybe transformer_xls need it?
LM_MAX_GRAD_NORM = 0.25 # VPT says 5.0, transfoXL says 0.25. We will basically c



def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def save_hidden_states_VPT(video_ids, hidden_state, saved_hidden_states):
    # Unpack the hidden states
    [(hidden_state_1, (hidden_state_2a, hidden_state_2b))] = hidden_state
    hidden_state_1[:]=False

    
    # Iterate over the batch dimension
    for i in range(BATCH_SIZE):
        video_id = video_ids[i]
        #video_id=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
        
        # Get the hidden state for this video
        video_hidden_state = [(hidden_state_1[i].clone(), (hidden_state_2a[i].clone(), hidden_state_2b[i].clone()))]
        
        # Save the hidden state for this video
        saved_hidden_states[video_id] = video_hidden_state
    
    return saved_hidden_states

def load_hidden_states_VPT(video_ids:List[str], saved_hidden_states):
    assert isinstance(video_ids, list)
    assert(len(video_ids)==BATCH_SIZE)
    global policy

    B = BATCH_SIZE
    T = 128
    E = VPT_WIDTH
    
    # Initialize the hidden states
    hidden_state_1 = th.zeros([B, 1, T], dtype=th.bool).to(DEVICE)
    hidden_state_2a = th.zeros([B, T, E], dtype=DTYPE).to(DEVICE)
    hidden_state_2b = th.zeros([B, T, E], dtype=DTYPE).to(DEVICE)
    
    # Iterate over the batch dimension
    for b in range(B):
        video_id = video_ids[b]
        #video_id=video.split(',')[0] IF WE DO THIS, HIDDEN STATE IS PRESERVED EVEN WHEN SWITCHING BETWEEN DIFFERNT TRAJECTORY FILES FROM THE SAME VIDEO. THIS DEPENDS ON THE SECTIONS BEING LOADED IN ORDER, WHIH SI NOT DONE HERE. this may increase val performance anyway by increasing the number of times mems is reset
        
        # Check if a hidden state has been saved for this video
        if video_id in saved_hidden_states:
            # Get the saved hidden state for this video
            [(video_hidden_state_1, (video_hidden_state_2a, video_hidden_state_2b))] = saved_hidden_states[video_id]
            
            # Set the hidden state for this video
            hidden_state_1[b] = video_hidden_state_1
            hidden_state_2a[b] = video_hidden_state_2a
            hidden_state_2b[b] = video_hidden_state_2b
        else:
            # Get a new initial hidden state for this video
            print("VPT NEW VIDEO SEEN: ADD FRESH INIT. STATE", video_id)
            [(_, (video_hidden_state_2a, video_hidden_state_2b))] = policy.initial_state(1)
            
            # Set the initial hidden state for this video
            #print(video_hidden_state_1.shape)
            is_first_frame_true = th.zeros((1, 128), dtype=th.bool).to(DEVICE)
            is_first_frame_true[:,0]=True
            hidden_state_1[b] = is_first_frame_true
            hidden_state_2a[b] = video_hidden_state_2a
            hidden_state_2b[b] = video_hidden_state_2b

    return [(hidden_state_1, (hidden_state_2a, hidden_state_2b))]







def load_hidden_states_LM(video_ids, saved_hidden_states):
    global policy
    assert isinstance(video_ids, list)

    T = policy.net.LM.transformer.mem_len
    n_layers = policy.net.LM.transformer.n_layer
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

    T = policy.net.LM.transformer.mem_len
    n_layers = policy.net.LM.transformer.n_layer
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
    global XATNN_MEMLEN
    
    T = XATNN_MEMLEN + SEQ_LEN
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
    


















def LM_train():
    
    global eval_data_loader, policy, agent, DEVICE, s2, s1 # for eval function
    if th.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    s1 = th.cuda.Stream()
    s2 = th.cuda.Stream()

    lowest_val_loss = [float('inf')]*4


    ### ---------------------------- initialise dataset and training
    print('Language Cloning: starting data loaders')
    ## Data Loader init
    train_data_loader = DataLoader(
                                    dataset_dir=DATASET+'train/',
                                    n_workers=N_WORKERS,
                                    batch_size=BATCH_SIZE,
                                    F_SEQ_LEN=F_SEQ_LEN,
                                    LM_TIMEOUT_RATE=LM_TIMEOUT_RATE,
                                    LM_SILENCE_TOKEN=2,
                                    n_epochs=EPOCHS,
                                    start_time='rand')


# to keep evaluation simple, thuogh possibly less accurate, we will just evaluate against the same [batch size] sequences. since its on fairly long sequcnes given the XL nature, this should still givne enough data points for some kind of useful evaluation


    ### ---------------------- initialise LC agent
    print('LC: LOADING VLPT')
    ### VPT INIT
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(VPT_MODEL_FILE)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs, LM_type="transfo-xl-wt103", LM_TIMEOUT_RATE=LM_TIMEOUT_RATE, L_SEQ_LEN=L_SEQ_LEN, dtype=DTYPE, LM_ONLY=True)
    agent.load_weights(VPT_WEIGHTS_FILE)
    policy = agent.policy # = th.compile(agent.policy)

    #init=policy.initial_state(1)
    #print('INITAIL STATE LEN',len(init))
    #print('INITAIL STATE TYPE',type(init))
    #1/0

    #policy.net.LM.load_state_dict(th.load('TRAINING/LM_ONLY/VLPT_LM_400__LM.weights')) #@@@
    #policy.net.Xattn_VPT_LM.load_state_dict(th.load('TRAINING/LM_ONLY/VLPT_LM_400__Xattn_VPT_LM.weights'))  #@@@


    policy.eval()
    policy.net.w_embed_dropout.train() # activate ddropout between words and Xattn0
    policy.net.recurrent_layer.blocks[-1].dropout.train() # activates dropout between VPT 0 and Xatt0
    policy.net.Xattn_VPT_LM.train() # activate dropout inside Xattn0
    policy.net.LM.train() # activate dropout inside LM
    








    # --- DEFINE OPTIMIZER                                                            # dont optimize CNN section.
    LM_parameters = list(     set(policy.net.LM.parameters()).union(   set(policy.net.Xattn_VPT_LM.parameters()))     ) # need to treat LM more with more fragility than rest of model. weight decay mainly, since VPT is likely already overfitting but LM is getting completely new data. include LM input XATNN gate

    # -- FREEZE non-LM, non-VPT-LM-Xattn SECTIONS
    VPT_params = list( set(policy.parameters()) - set(LM_parameters) )
    for param in VPT_params:
        param.requires_grad = False


    #optimizer = th.optim.AdamW(params=LM_parameters, lr=VPT_LEARNING_RATE, weight_decay=LM_WEIGHT_DECAY},
    optimizer = th.optim.Adam(params=LM_parameters, lr=LM_LEARNING_RATE) # normal adam as used in https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/train.py
    #optimizer.load_state_dict(th.load('TRAINING/LM_ONLY/_400_.optim')) #@@@

    lr_schedule = CosineAnnealingWarmupRestarts(optimizer,
                first_cycle_steps=max_train_steps,
                cycle_mult=1.0,
                max_lr=LM_LEARNING_RATE,  #@ WARNING: this sets both VPT and LM learnig rates to the same. For now this is okay because they are the same anyway, but this will need modifying if different learning rates are used in the end
                min_lr=0,
                warmup_steps=warmup_steps,
                gamma=1.0)
    #lr_schedule.load_state_dict(th.load('TRAINING/LM_ONLY/_400_.lrschedule')) #@@@
    
    
    
    
    # --------------------------- start training loop
    print('LC: MAIN LOOP:')
    saved_hidden_states_VPT = {} # this is so that, despite workers>batch size and therefore video not being streamed in perfect order across batches, we can keep track of VPT_hidden_stae and LM_hidden state
    saved_hidden_states_LM = {}  # same^
    saved_hidden_states_Xattn1 = {}
    saved_hidden_states_Xattn2 = {}
    is_first_frame = th.zeros((BATCH_SIZE, L_SEQ_LEN*LM_TIMEOUT_RATE), dtype=th.bool).to(DEVICE)
    current_video_group_id = [0]*BATCH_SIZE
    start_time = time.time()
    loss_sum=[]
    val_loss_sum=np.zeros([0,2])
    gates=np.zeros([0,2])
    # get multiple steams of 10 minutes* video across multiple batches. continue until (to ensure lanauge model sees far back langauge)
    for batch_i, (video_group_id, subseq_ids, batch_frames, batch_words, _, finished_videos) in enumerate(train_data_loader):



        

        # ----------------------------------------- EVALUATION
        if batch_i%EVALUATION_RATE == 0:
            print("## ---------------------------------- - EVAL - ---------------------------------------", batch_i)
            eval_data_loader = DataLoader(
                dataset_dir=DATASET+'valid/',
                n_workers=EVAL_BATCH_SIZE,
                batch_size=EVAL_BATCH_SIZE,    # SINCE EVAL DOESNT DO COPMLICATED STATE-SAVING, ITS CRUCIAL FOR mem INTEGRIDY THAT VIDEAS ARE STREAMED ACROSS TEH SAME SUB-BATCH SAMPLE INDEX EVERY BATCH, AND THAT A VIDEO IS IN EVERY BATCH UNTIL IT ENDS, ONLY THEN DO WE ADD A NEW VIDEO FROM THE ONES WE ASTARTED WITH
                F_SEQ_LEN=L_SEQ_LEN*LM_TIMEOUT_RATE, 
                n_epochs=1,
                max_subseqs_per_traj=10,
                start_time=120)
            LM_eval_loss, noised_LM_eval_loss = BLC_evaluate()
            del eval_data_loader
            
            val_loss = np.asarray([[LM_eval_loss, noised_LM_eval_loss]])
            val_loss_sum = np.concatenate([val_loss_sum, val_loss])

            # --- plot val loss
            plt.plot(val_loss_sum[:,0], color='blue')
            try:
                os.remove('TRAINING/LM_ONLY/val_loss_graph_.png')
            except: 
                pass
            plt.savefig('TRAINING/LM_ONLY/val_loss_graph_.png')
            plt.clf()

            # ---  plot performance difference between matched words/frames and unmatched
            plt.plot(val_loss_sum[:,1]-val_loss_sum[:,0], color='black') # if NN learning to use frames properly, then noised_loss>val_loss, so graph goes up
            try:
                os.remove('TRAINING/LM_ONLY/noised_diff_graph_.png')
            except: 
                pass
            plt.savefig('TRAINING/LM_ONLY/noised_diff_graph_.png')
            plt.clf()


            line=str("Eval: LM_loss: {0}, \nnoised: {1} \nXattnGate(0,1):{2}".format(
                str(LM_eval_loss), 
                str(noised_LM_eval_loss),
                str(agent.policy.net.Xattn_VPT_LM.alpha_xattn.item())+','+str(agent.policy.net.Xattn_VPT_LM.alpha_dense.item())
            ))
            with open('TRAINING/LM_ONLY/training_log_val','a') as file:
                file.write(line)
                
            # save a model if ALL losses are lower
            save_model=True
            for loss_lowest, loss_new in zip(lowest_val_loss, [LM_eval_loss, noised_LM_eval_loss]):
                if loss_lowest < loss_new:
                    save_model = False
                        
            if save_model:
                print("#----------------------- BEST VAL LOSS! SAVING!")
                # SAVE MODEL WEIGHTS
                lowest_val_loss = [LM_eval_loss, noised_LM_eval_loss]
                output_path = '/'.join(OUTPUT_WEIGHTS.split('/')[0:-1])+'/'
                output_bonus = '_'+str(batch_i)+'_'
                output_name = '.'.join(OUTPUT_WEIGHTS.split('/')[-1].split('.')[0:-1])+output_bonus
                th.save(policy.net.LM.state_dict(), output_path+output_name+'_LM.weights')             # save LM weights
                th.save(policy.net.Xattn_VPT_LM.state_dict(), output_path+output_name+'_Xattn_VPT_LM.weights')                # save VP->LM Xattn weights
                
                # ALSO SAVE OPTIMIZER AND LEARNING_RATE_SCHEDULER STATES
                th.save(optimizer.state_dict(), output_path+output_bonus+'.optim')
                th.save(lr_schedule.state_dict(), output_path+output_bonus+'.lrschedule')
            print('## ---------------------------------- - TRAIN - ---------------------------------------"')
        
        
        
        
        



        # ----------------------------------------- TRAIN ---------------------------------
        policy.zero_grad(set_to_none=True)
        th.cuda.empty_cache()
        
        ### --- LOAD CORRECT VID MEMS
        #print('load_hid..')
        VPT_state = load_hidden_states_VPT(video_group_id, saved_hidden_states_VPT)
        LM_state = load_hidden_states_LM(video_group_id, saved_hidden_states_LM)
        Xattn1_state = load_hidden_states_Xattn(video_group_id, saved_hidden_states_Xattn1, SEQ_LEN=F_SEQ_LEN, E=VPT_WIDTH)
        Xattn2_state = load_hidden_states_Xattn(video_group_id, saved_hidden_states_Xattn2, SEQ_LEN=L_SEQ_LEN, E=1024)
        ### lookingat current videos in the batch, retreive the appropriate previously calculated hiddens states from teh last batch of frames for that video.
        # this is  complicated because LM and VPT have their own states and each expects them as a single batch and ahs different ways of initialising athem andindcating
        # how memories should be reset. the save and load functiosn to this - de-batching them, separatin them out by video_name, storing, 
        # checking new video_ids at this particular batch, fwetching teh appropriate hidden states and stitching them back together into
        # a batch where the appropriate mems are in the correct order for the to be applied to the correct video.
        # we also modify the data_loader to indicate when videos end so that we dont keep piling more states onto the mems storage, but delete them whe
        # theyre not needed anymore


        ### ------------ FORMAT INPUT      
        # format words
        x_words, y_words = batch_words['input_ids'], batch_words['labels']
        x_words=th.from_numpy(x_words).to(DEVICE)
        y_words=th.from_numpy(y_words).to(DEVICE)
        # format input frames
        batch_frames['img'] = th.from_numpy(batch_frames['img']).to(DTYPE).to(DEVICE)
        # format action labels



        ## ----------------- VLPT MODEL FORWARD PASS
        # PREDICT VLPT (input frames and paired language tokens). Get output VPT actions, and LM loss
        _, _, _, VPT_state, LM_state, LM_loss, Xattn1_state, _ = policy.get_output_for_observations(
            ob_words=x_words,
            ob_frames=batch_frames,
            VPT_state=VPT_state,
            first=is_first_frame,
            LM_state=LM_state,
            LM_labels=y_words,
            LM_ONLY=True,
            Xattn1_state=Xattn1_state,
            Xattn2_state=Xattn2_state)


        
        # ------------------------- BACKWARD PASS
        LM_loss.backward()
        th.nn.utils.clip_grad_norm_(LM_parameters, LM_MAX_GRAD_NORM)
        optimizer.step()
        lr_schedule.step()
        policy.zero_grad(set_to_none=True)
        th.cuda.empty_cache()
        

        # Make sure we do not try to backprop through sequence in future iterations
        LM_state = tree_map(lambda x: x.detach(), LM_state)
        VPT_state = tree_map(lambda x: x.detach(), VPT_state)
        Xattn1_state = Xattn1_state.detach()

        # save hidden states from these videos for next time they show up. print('save_hid..')
        save_hidden_states_VPT(video_group_id, VPT_state, saved_hidden_states_VPT)
        save_hidden_states_LM(video_group_id, LM_state, saved_hidden_states_LM)
        save_hidden_states_Xattn(video_group_id, Xattn1_state, saved_hidden_states_Xattn1)

        # --- free up hidden states whose videos have ended (i.e. fix memory leak in original VPT github)
        for video in finished_videos:
            if video in saved_hidden_states_VPT:
                print("video ended:",video," cleaning up hidden state...")
                saved_hidden_states_VPT.pop(video)
                saved_hidden_states_LM.pop(video)









        # ------------------------------------------ LOSS REPORTING-------------------
        os.chdir('/content/drive/MyDrive/_DISSERTATION/')
        loss = LM_loss.mean().item()
        loss_sum.append(loss)
        gates_now = np.asarray([[   abs(agent.policy.net.Xattn_VPT_LM.alpha_xattn.item()),
                                    abs(agent.policy.net.Xattn_VPT_LM.alpha_dense.item())   ]])
        gates=np.concatenate([gates,gates_now],axis=0)
        
        if batch_i%LOSS_REPORT_RATE==0:
            print('logging progress...   ex text:', x_words[0,0:20])
            time_since_start = time.time() - start_time
            
            #plot loss
            plt.plot(loss_sum, color='blue')
            try:
                os.remove('TRAINING/LM_ONLY/loss_graph_.png')
            except:
                pass
            plt.savefig('TRAINING/LM_ONLY/loss_graph_.png')
            plt.clf()
            plt.close()
            
            #plot gates
            plt.plot(gates[:,0], color='darkred')
            plt.plot(gates[:,1], color='red')
            try:
                os.remove('TRAINING/LM_ONLY/gates_graph_.png')
            except:
                pass
            plt.savefig('TRAINING/LM_ONLY/gates_graph_.png')
            plt.clf()
            plt.close()
            
            # record training progress - so that if it crashes part way through, we can re-try training and resume from the same spot (actually this implementation dumps the rest of the video and we start at the next one.)
            with open(TRAINING_PROGRESS,  'a') as progress_file:
                line=str(batch_i)+str(video_group_id)+str(subseq_ids)
                progress_file.write(line)
            line=str(   "Eval: Time:{0}, LM_loss: {1}, \nXattnGate(0,1):{2}".format(
                        str(time_since_start),
                        str(LM_loss),
                        str(agent.policy.net.Xattn_VPT_LM.alpha_xattn.item())+','+str(agent.policy.net.Xattn_VPT_LM.alpha_dense.item()),
                    ))
            with open('TRAINING/LM_ONLY/training_log','a') as file:
                file.write(line+'\n')







        
        # reset losses    
        print(' --- FINISHED BATCH', video_group_id, subseq_ids, finished_videos, batch_i, LM_loss.mean().item() ) #
        LM_loss, VPT_loss, BLC_loss, KL_divergence=0,0,0,0














def BLC_evaluate():    
    
    
    with th.no_grad():
        global agent, eval_data_loader, policy, s2, s1
        
        # put VLPT into testing mode
        policy.eval()
    
        # we dont want to disrupt internal states of training during eval so we use fresh ones
        eval_current_video_group_id = 0
        eval_LM_loss =th.tensor(0.).to(DEVICE)
        noised_eval_LM_loss =th.tensor(0.).to(DEVICE)
    
        eval_is_first_frame = th.zeros((EVAL_BATCH_SIZE, EVAL_SEQ_LEN*LM_TIMEOUT_RATE), dtype=th.bool).to(DEVICE)
        num_batch=0
        for batch_i, (eval_video_group_id, eval_subseq_ids, eval_batch_frames, eval_batch_words, _, _) in enumerate(eval_data_loader): 
            # dont need to bother with proper state saving, we just need that during training for batch diversity for training stabilisation
            num_batch+=1
            if eval_video_group_id != eval_current_video_group_id:  
                eval_current_video_group_id = eval_video_group_id # we dont want to modify internal states of training during eval so we use fresh ones
                
                eval_VPT_state = policy.initial_state(EVAL_BATCH_SIZE)
                eval_LM_state = None
                noised_eval_VPT_state = policy.initial_state(EVAL_BATCH_SIZE)
                noised_eval_LM_state = None

                eval_Xattn1_hidden_state = th.zeros([EVAL_BATCH_SIZE,XATNN_MEMLEN+F_SEQ_LEN,VPT_WIDTH], dtype=DTYPE).to(DEVICE)
                eval_Xattn2_hidden_state = th.zeros([EVAL_BATCH_SIZE,XATNN_MEMLEN+L_SEQ_LEN,1024], dtype=DTYPE).to(DEVICE)
                
                noise_eval_Xattn1_hidden_state = th.zeros([EVAL_BATCH_SIZE,XATNN_MEMLEN+F_SEQ_LEN,VPT_WIDTH], dtype=DTYPE).to(DEVICE)
                noise_eval_Xattn2_hidden_state = th.zeros([EVAL_BATCH_SIZE,XATNN_MEMLEN+L_SEQ_LEN,1024], dtype=DTYPE).to(DEVICE)


            ### ------------- FORMAT INPUT DATA
            # format input frames
            eval_batch_frames['img'] = th.from_numpy(eval_batch_frames['img']).to(DTYPE).to(DEVICE)
            #eval_batch_frames['img'] = eval_batch_frames['img'].to(DEVICE)
            # format input/label words
            x_words, y_words = eval_batch_words['input_ids'], eval_batch_words['labels']
            x_words=th.from_numpy(x_words).to(DEVICE)
            y_words=th.from_numpy(y_words).to(DEVICE)

            noise_x_words = x_words.clone()
            noise_x_words = noise_x_words.roll(1,dims=0)
            noise_y_words = y_words.clone()
            noise_y_words = noise_y_words.roll(1,dims=0)



            ### ----------- MODEL FORWARD: pass frames to [VPT CNN, VPT transfo_1] and pass result to LM, then predict words
            th.cuda.empty_cache()
            _, _, words_out, eval_VPT_state, eval_LM_state, e_LM_loss, eval_Xattn1_hidden_state, _ = policy.get_output_for_observations(
                        ob_words=x_words,
                        ob_frames=eval_batch_frames,
                        VPT_state=eval_VPT_state,
                        LM_state=eval_LM_state,
                        LM_labels=y_words,
                        first=eval_is_first_frame,
                        LM_ONLY=True,
                        Xattn1_state=eval_Xattn1_hidden_state,
                        Xattn2_state=eval_Xattn2_hidden_state) # never actually used bc LM only
                #print('1', str(e_LM_loss.mean().item()),  str(eval_batch_frames['img'][0,0].mean().item()), str(words_out.mean().item()))
          
      
            # NOISED LM INPUT: LM will improve even if its not actually using VPT input, since it is learning to use adapt from wt103 pretraining to silence tokens and saying minecraft-related text, regardless of input. We therefore need to test how well it performs when the VPT input is randomised (to test fairly and not unfairly do OOD, we test with the frames inputs swapped)
	           ### -----------.> feed batch of input sequences (frames and paired language tokens) to agent and get output action and
            _, _, noised_words_out, noised_eval_VPT_state, noised_eval_LM_state, noised_e_LM_loss, noise_eval_Xattn1_hidden_state, _ = policy.get_output_for_observations(
                    ob_words=noise_x_words,
                    ob_frames=eval_batch_frames,
                    VPT_state=noised_eval_VPT_state,
                    LM_state=noised_eval_LM_state,
                    LM_labels=noise_y_words,
                    first=eval_is_first_frame,
                    LM_ONLY=True,
                    Xattn1_state=noise_eval_Xattn1_hidden_state,
                    Xattn2_state=noise_eval_Xattn2_hidden_state) # never actually used bc LM only
            #print('2', str(noised_e_LM_loss.mean().item()),     str(eval_batch_frames['img'][0,0].mean().item()), str(noised_words_out.mean().item()))

            th.cuda.empty_cache()

            # calculate loss
            eval_LM_loss += e_LM_loss
            noised_eval_LM_loss += noised_e_LM_loss
            
            print("LC_EVAL: batch done!", eval_video_group_id, eval_subseq_ids, e_LM_loss.item(), noised_e_LM_loss.item(), noised_e_LM_loss-e_LM_loss)

            th.cuda.empty_cache()
        
        
        
        del noised_eval_VPT_state, eval_VPT_state
        
        eval_LM_loss /= num_batch
        noised_eval_LM_loss /= num_batch
    
        # return VLPT to training mode
        policy.eval()
        policy.net.w_embed_dropout.train() # activate ddropout between words and Xattn0
        policy.net.recurrent_layer.blocks[-1].dropout.train() # activates dropout between VPT 0 and Xatt0
        policy.net.Xattn_VPT_LM.train() # activate dropout inside Xattn0
        policy.net.LM.train() # activate dropout inside LM


        print("LC_AVG: ", noised_eval_LM_loss-eval_LM_loss )
        return eval_LM_loss.cpu().numpy(), noised_eval_LM_loss.cpu().numpy()
        

    # agent estimate 10 video sequence batches of 512 with same tgt_len and mem_len timeout








if __name__ == "__main__":
    #parser = ArgumentParser()
    #parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    #parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    #parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    #parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    #args = parser.parse_args()
    #BLC_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
    LM_train()
