# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.
# NOTE: This is _not_ the original code used for VPT!
#       This is merely to illustrate how to fine-tune the models and includes
#       the processing steps used.                                               @FIX THIS to run on 4x3090 = 96GB:

# This will likely be much worse than what original VPT did:
# we are not training on full sequences, but only one step at a time to save VRAM.










# TODO: 
"""

--- FIX
- Edit DataLoader: load transcipt.file and give out SEQ_LEN frames, SEQ_LEN actions and all words in episode
- Data Loader: (if doesnt fit in batch,m just discard batch, end episode early) (send episode_ended signal to clear hidden state when done. How to clear hidden state? how hidden staet batch???)


--- OPTIMIZE
- put VPT and LM on different GPU
- DataParrallel and OGtransfoxlDataParallel.  - maybe just wrap whol model in OGtransfoxlDataParallel?
- test:
    IDM prediction on labelled dataset -> store
    VPT prediction on labelled dataset
    VPT prediction on stored IDM labels

    IDM prediction on labelled dataset (YT240p) -> store
    VPT prediction on labelled dataset (YT240p)
    VPT prediction on stored (YT240p) IDM labels

    IDM prediction on labelled dataset (YT480p) -> store
    VPT prediction on labelled dataset (YT2480p)
    VPT prediction on stored (YT480p) IDM labels
- VLPT: fuse hidden states in and out of lm - allow LM to se VPT mem in layer below. allow VPT 3 to see LM mem
- currently, TransformerXL's softmx classification layer only either support gettings the loss or predicting wor dprobabilities, not both. This means that it must be passed through twice in ordre to get both, which is slow. TO optimize, finda  way to sample AdaptiveLogSoftmax while getting loss.

- check PaLI/Flamingo paper for training procedure: KL, losses...


--- WORDS
- code YT_link -> audio collector
- code audio -> transcript + timestamps
- create script to take audio from video, check if WPM is good (180 - 240), then save audio somewhere safe if good, along with video link to a file.

--- VIDEO
- FOR DENSE LANGUAEG USE 'TUTORIAL' KEYWORD
- manually label 8000 frames spam/ham
- train SVM+frozen CLIP as spamham
- create script to select a video, select a few seconds randomly, download whole video if 80% gathered frames are clean

--- INITIAL TESTING
- Finetune pure VPT on collected video dataset using VLPT codebase with langauge model set as not active.
- on collected video audio data with NULLS filtered out, check how often silence tokens appear (some balance between peak and average) and set D appropriately
- Finetune Transfo_XL on collected words dataset


--- EVALUATION
check different way of looking at learn representations: attention in LM over time/space? attention in VPT to VPT1/ LM?
"""




#NOTE:
"""
------ HYPERPARAMS:

###DATASET
episode length = first 10 mins| based on: openAI used first  mins for 'early game' finetuning, longer for others. Not long anough for long term langauge stuff.

### XLRECURRENCE
LM_tgt_len = 256 |227 | based on: in order for LM to see past 10 minutes (assuming 50ms per frame, mem_len = tgt_len(?), ). 10 minutes good, needs  to see algnauge, neeed to keep LM intact: it was trained on mem = 384 tgt=384, so we need to get as close as possible. 256 is close, although we still get silence tokens I dont think results should be clear at these numbers and I think higher is probably too expensive. transfoxl was trained even with tgt=128 wiht useful results
LM_mem_len = 256 | 227 | based one: TransformerXL paper uses same memlen as tgtlen during training, so using same as tgt^ here (assuming it is for equal learning progress on both mechanisms). During eval can be made much longer. (Maybe test same with VPT?). 
VPT_tgt_len = 128 | based on: original VPT paper, minimise differences, minimise compute
VPT_mem_len = 128 | based on: Given `prev` keys from cache, and `new` keys,
            returns (cache, full), where
            - cache goes into the output state, length chosen so that on the
                next timestep, there are enough cached timesteps to get the full
                context of lenth self.maxlen.


# JOINT MODEL:



### LEARNING PARAMS:
learning rate:   | based on: PaLI lr = 
learning rate warmup: 5000 | based on: cant remember
learning rate cooldown: cosine until last step
num steps = 
how long to train: how long can I run A100? $10 = 100 creds = 7.5 compute hours. how long does a single batch take? idk slowest possible = 

"""




from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np
from lib.data_parallel import BalancedDataParallel

from agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from lib.tree_util import tree_map

if th.cuda.is_available():
    device = th.device('cuda')
    print("USING CUDA")
else:
    device = th.device('cpu')
    print("USING CPU")

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts



""" both VPT and LM are TransformerXL. Remember that the incoming data must reflect this - each batch has samples of videos in an order. each sequcen is in order (obviously, in terms of how frames  ioni teh sequnce are ordered) and has an ID. 
After each batch they both save a fraction of their internal outputs from that batch and bring it to the next batch so they can bring past memories to the continuation of teh sequences in the repvious batch.
This means that every batch must consist of teh same videos in the previous batch (but the next frames in time obviously) and they must be in the same order within that batch

 batch 1       batch 2     and so on
a,b,c,d,e    f,g,h,i,j
q,r,s,t,u    v,w,x,y,z      ...
g,h,i,j,k    l,m,n,o,p

How long the input sequnces are and how far back tokens the recurrent mechanism can reach back determine what fraction of tokens are processed by either method
we wwant them even, so inptus equnce length = mem size

Since we have many videos, we need to organise the taining such that at a point the memories are reset and we change which videos are being inputted across batches.
We need to make sure hidden states are managed so that everything is coherent - we dont wat to give the memories of one episode to the agent when is working ona  different episode.
This work also avoids passing one at the beginning of training to next ones since this trains the agent to continue using its original old hiddenr representations,
so video sequences are kept continuous and all made to end when the videos are over (they are all trimmed to the same length), at which point all memories for VPT and LM are reset.
"""





scaler = th.cuda.amp.GradScaler()



DATASET = '../Data Gathering/DATASET'
EPOCHS = 20
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)       # -------- ah, just what I'm looking for!
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 8
SEQ_LEN = 512  # WE WANT 10 minutes of langauge that LM can look back on:  ( seq_len*18*D*50 ) / (1000*60)      @seq_len=256, D=1 this is about 4 mins. Sequence length needs to be chosen so that when VPT predict this many frames it proved LM with tgt_len inputs to predict with. adjusting this threfore adjsuts tgt_len and how far back in time LM can see during training.
DEVICE = "cuda"
LOSS_REPORT_RATE = 100
EVALUATION_RATE = 100

# PaLI:
#     train: 0.02Adafactor    fintune:0.001adafactor
# LM: train:0.01adafactor  finetune:0.001adafactor
# V:  train:0.0008

# Flamingo
# 0.0001 adamw
# LM 0.0001 adamw

#BCT
#VPT: 0.002147Adam  finetune:0.000181Adam, weight decay 

# transfo_xl:
#LM:  0.00025Adam. Weight decay=PyTorchdefault=0 . suggested = 0.01

VPT_LEARNING_RATE = 0.00025 # VPT paper sugests 0.000181: [we are training to a very different task], [VPT uses linear learning rate decay], [] # to keep the LM intact I dont want to go higher tha its peak laerning rate. finetuning a multimdodal LM with the same peak lr seems ok according to PaLI,Flamingo but they also train on other tasks, maybe just keep some minecraft data for langauge training?
VPT_WEIGHT_DECAY = 0.039428 # VPT weigh decay. transfoxl weight decay is 
VPT_MAX_GRAD_NORM = 5.0 # VPT says 5.0, transfoXL says 0.25. We will basically c

LM_LEARNING_RATE = 0.00025 # both LRs are actually set to be the same by the scheduler anyway. double check if they should be different and adjust accordingly. These learning rates are tiny, but maybe transformer_xls need it?
LM_WEIGHT_DECAY = 0.01 # avoid LM helping VPT to overfit
LM_MAX_GRAD_NORM = 0.25 # VPT says 5.0, transfoXL says 0.25. We will basically c


VPT_MODEL_FILE = '2x.model'
VPT_WEIGHTS_FILE = 'rl_from_early_game'
TRAINING_LOG_FILE = 'training_log.txt'
# VPT model automatically downloads transfo_xl weights from HuggingFace and uses those for LM. If weights include the LM it should be overwritten though?

num_videos = 6000
max_train_steps = (num_videos*600000)/(SEQ_LEN*BATCH_SIZE)    # 10 mins per video = 600000 ms -> 4687 chunks of 128 frames. want 1000 hours video = 60,000 minutes = 6,000 videos of 10 minutes each
warmup_steps = min(1000, max_train_steps*0.03) # warmup should be very short since the transformers are pretrained # PaLI uses 1k warmup steps, obviously dont want to do more




def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs







def BLC_train(data_dir, in_model, in_weights, out_weights):
    global eval_data_loader, para_model, agent # for eval function
    lowest_val_loss = float('inf')

   ### ---------------------- initialise BLC agent
    ### VPT INIT
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(VPT_MODEL_FILE)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs).to(DEVICE)
    agent.load_weights(VPT_WEIGHTS_FILE)
    policy = agent.policy
    trainable_parameters = agent.policy.parameters()

    if args.gpu0_bsz >= 0:
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, policy, dim=1).to(device)
    else:
        para_model = policy #th.nn.DataParallel(policy, dim=1).to(device) # I think this fails? HF said this is bugged, right?
    #initiliase VPT memories to empty
    VPT_state = None

    ### LM INIT
    # initialise LM memories to empty
    LM_state = None

    # define optimizer # we have slightly difference hyperparameters for VPT and LM. This is partially based on the fact that their respective papers use quite different values,
    # the difference in architecture due to that,
    # and because VPT is very likely to overfit: it has been finetuned on rl so its weights are departed from its video/behavioural-cloning pretraining (although not that much ebcause of the KL-loss term), but we are likely going to get data from 
    # videos that that paper pulled since it trained on so much data. We will be training on some stuff that wasnt in the BC finetuning since we are using the early-game model which only used first 5 mins are we are using first 10 for lagnauge input purposes. However, the foundation model still mght have seen all of this.
    # regardless, VPT is advised to have and likely should have a high weight decay while the LM was trained without one, and adding such a high weight decay would likely destroy it. We cannot rely on KL-loss since the LM is being pruposely trained to behave very differently to its original - the KL loss target would only produce useless noise du to teh Xattn inpuot adn silence tokens, so it is just bad for performance to use it as a reference
    VPT_parameters=[]
    LM_parameters = para_model.net.LM.parameters() # need to treat LM more with more fragility than rest of model. weight decay mainly, since VPT is likely already overfitting but LM is getting completely new data. include LM input XATNN gate
    for param in para_model.net.Xattn_VPT_LM.parameters():
        LM_parameters.append(param)
    for param in trainable_parameters:
        if not (param in LM_parameters): 
            VPT_parameters.append(param) # include LM output Xattn
    optimizer = th.optim.AdamW(
                [{'params': VPT_parameters, 'lr': VPT_LEARNING_RATE, 'weight_decay':VPT_WEIGHT_DECAY},
                {'params': LM_parameters, 'lr': LM_LEARNING_RATE, 'weight_decay':LM_WEIGHT_DECAY}])

    lr_schedule = CosineAnnealingWarmupRestarts(optimizer,
                first_cycle_steps=max_train_steps,
                cycle_mult=1.0,
                max_lr=LM_LEARNING_RATE,  #@ WARNING: this sets both VPT and LM learnig rates to the same. For now this is okay because they are the same anyway, but this will need modifying if different learning rates are used in the end
                min_lr=0.0,
                warmup_steps=1000,
                gamma=1.0)




    ### ---------------------------- initialise dataset and training
    ## Data Loader init
    train_data_loader = DataLoader(
    dataset_dir=DATASET+'/train',
    n_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    seq_len = SEQ_LEN,
    n_epochs=EPOCHS)

    # load examples 128*40 frames
    # load example actions
    # load example actions
    ## Data Loader init
    eval_data_loader = DataLoader(
    dataset_dir=DATASET+'/valid',
    n_workers=8,
    batch_size=8, # to keep evaluation simple, thuogh possibly less accurate, we will just evaluate against the same [batch size] sequences. since its on fairly long sequcnes given the XL nature, this should still givne enough data points for some kind of useful evaluation
    seq_len=SEQ_LEN,
    n_epochs=EPOCHS)




    # --------------------------- start training loop
    is_first_frame = th.zeros((BATCH_SIZE, SEQ_LEN), dtype=th.bool).to(DEVICE)
    current_video_group_id = 0
    current_batch = 0
    start_time = time.time()
    # get multiple steams of 10 minutes* video across multiple batches. continue until (to ensure lanauge model sees far back langauge)
    for batch_i, (batch_frames, batch_words, batch_actions, video_group_id) in enumerate(train_data_loader):

        # zero grads
        for param in trainable_parameters:     #https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            param.grad = None
        lr_schedule.step()

        with th.cuda.amp.autocast(device_type='cuda', dtype=th.float16): #https://pytorch.org/docs/master/notes/amp_examples.html#gradient-clipping
            # multiple batches from the same video group will occur in a row. Dont reset memory until a different video group comes along.
            if current_video_group_id != video_group_id:
                is_first_frame[:,...] = True
                para_model.LM.reset_mem()
                current_video_group_id = video_group_id


            ### --- FORMAT INPUT      
            # format input frames
            x_frames = agent._video_obs_to_agent(batch_frames['img'])
            # format input/label words
            x_words, y_words = agent._words_to_agent(batch_words['token_ids'], batch_words['ms'])
            #format action labels
            #actions_formatted = th.zeros([BATCH_SIZE, SEQ_LEN])        # NULL ACTIONS NOT REMOVED: this is probably fine because we still get rid of al lot of null actions (any that happend with no paired word. thjis is done to maintain language integrity and timings). We can mask NULL actions now though. Probably should to maintain comparability to VPT, but this will proabbly hurt performance - VPT suggests getting rid of most but not all NULL actionsm which this probably does. IDK
            action_labels = agent._env_action_to_agent(batch_actions, to_torch=True, check_if_null=False) 
            #actions_formatted[b,t] = action


            ### --- PREDICT (input frames and paired language tokens). Get output VPT actions, and LM loss
            pi_distribution, _, VPT_state, LM_state, LM_loss = para_model.get_output_for_observation(
                                                                                                        ob_words=x_words,
                                                                                                        ob_frames=x_frames,
                                                                                                        VPT_state_in=VPT_state,
                                                                                                        context=is_first_frame,
                                                                                                        LM_state=LM_state,
                                                                                                        LM_active_timestep=True,
                                                                                                        last_future_words=None,
                                                                                                        LM_labels=y_words)



            # --- optimize model
            BLC_loss = VPT_loss + LM_loss
        scaler.scale(BLC_loss).backward()
        scaler.unscale_(optimizer)
        # clip LM gradient to a smaller clip value
        th.nn.utils.clip_grad_norm_(VPT_parameters, VPT_MAX_GRAD_NORM)
        th.nn.utils.clip_grad_norm_(LM_parameters, LM_MAX_GRAD_NORM)
        #for parameter in para_model.net.LM.parameters():
        #    if parameter.requires_grad:
        #        th.nn.utils.clip_grad_norm_(parameter, MAX_GRAD_NORM_LM)
        scaler.step(optimizer)
        scaler.update()



        # # Make sure we do not try to backprop through sequence in future iterations
        # (fails with current accumulation). 
        # LM does this internally automatically.
        VPT_state = tree_map(lambda x: x.detach(), VPT_state)
        VPT_loss  = -para_model.get_logprob_of_action(pi_distribution, action_labels)



        # --- keep track of model loss and val loss, save model if new best val_loss (on all 3 tasks)
        loss_sum += VPT_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            line = str(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            with open('training_log.txt','a') as file:
                file.write(line)
            loss_sum = 0
        if batch_i % EVALUATION_RATE == 0:
            LM_eval_loss, VPT_eval_loss, BLC_loss = BLC_evaluate()
            line=str("Eval: LM_loss, VPT_loss, BLC_loss: %s" %(LM_eval_loss, VPT_eval_loss, BLC_loss))
            with open('training_log.txt','a') as file:
                file.write(line)
            # save a model if ALL losses are lower
            save_model=True
            for loss_lowest, loss_new in zip(lowest_val_loss, [LM_eval_loss, VPT_eval_loss, BLC_loss]):
                if loss_lowest < loss_new:
                    save_model = False
            if save_model:
                state_dict = para_model.state_dict()
                th.save(state_dict, out_weights+str(batch_i))





def BLC_evaluate():
    with th.no_grad():
        global agent, eval_data_loader, para_model


        is_first_frame = th.zeros((BATCH_SIZE, SEQ_LEN), dtype=th.bool).to(DEVICE)

        # we dont want to disrupt internal states of training during eval so we use fresh ones
        eval_LM_state = None
        eval_VPT_state = None
        for batch_i, (batch_frames, batch_words, batch_actions, video_group_id) in enumerate(eval_data_loader):


            ### ------------- format input from data loader to agent        
            # format input frames
            x_frames = agent._video_obs_to_agent(batch_frames['img'])
            # format input/label words
            x_words, y_words = agent._words_to_agent(batch_words['token_ids'], batch_words['ms'])
            #format action labels
            #actions_formatted = th.zeros([BATCH_SIZE, SEQ_LEN])        # NULL ACTIONS NOT REMOVED: this is probably fine because we still get rid of al lot of null actions (any that happend with no paired word. thjis is done to maintain language integrity and timings). We can mask NULL actions now though. Probably should to maintain comparability to VPT, but this will proabbly hurt performance - VPT suggests getting rid of most but not all NULL actionsm which this probably does. IDK
            action_labels = agent._env_action_to_agent(batch_actions, to_torch=True, check_if_null=False) 
            #actions_formatted[b,t] = action

            ### ----------- feed batch of input sequences (frames and paired language tokens) to agent and get output action and 
            pi_distribution, _, VPT_state, LM_state, LM_loss = para_model.get_output_for_observation(
                                                                                                        ob_words=x_words,
                                                                                                        ob_frames=x_frames,
                                                                                                        LM_labels=y_words,
                                                                                                        VPT_state_in=eval_VPT_state,
                                                                                                        LM_state=eval_LM_state,
                                                                                                        context=is_first_frame,
                                                                                                        LM_active_timestep=True)

            LM_loss = LM_loss
            VPT_loss  = -agent.policy.get_logprob_of_action(pi_distribution, action_labels)
            BLC_loss = VPT_loss + LM_loss

            return LM_loss, VPT_loss, BLC_loss

    # agent estimate 10 video sequence batches of 512 with same tgt_len and mem_len





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    BLC_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)





"""
VPT240M_parameters = load_model_parameters("2x.model")
VPT240M_parameters = 

attention_heads': 16,
  'attention_mask_style': 'clipped_causal',
  'attention_memory_size': 256,
  'diff_mlp_embedding': False,
  'hidsize': 2048,
  'img_shape': [128, 128, 3],
  'impala_chans': [16, 32, 32],
  'impala_kwargs': {'post_pool_groups': 1},
  'impala_width': 8,
  'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1},
  'n_recurrence_layers': 4,
  'only_img_input': True,
  'pointwise_ratio': 4,
  'pointwise_use_activation': False,
  'recurrence_is_residual': True,
  'recurrence_type': 'transformer',
  'timesteps': 128,
  'use_pointwise_layer': True,
  'use_pre_lstm_ln': False},

  """