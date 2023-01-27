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
- Do: audio -> tokens + tokens ms -> save to file
- Edit DataLoader: load transcipt.file
- add val_loss iteration every 100 batches: VLPT_val_loss, LM_val_loss, LM_wt103_loss NOTE: scraping web, will probably get some same videos as VPT paper, may result in overfitting. we are using RL finetuned model o hopefully that made it forget but just be careful. Might need dropout but try first without. also remove dropout form LM, wont train for many epochs, can get a LOT of data
- save model every 10000 batches: for MineRL assessment through time.
- Data Loader: (if doesnt fit in batch,m just discard batch, end episode early) (send episode_ended signal to clear hidden state when done. How to clear hidden state? how hidden staet batch???)
- IGNORE: just set D=1 and progress and dont worry about batch and grad accum rates and tgt_len differences compensation in batch handling arghh

--- OPTIMIZE
- put VPT and LM on different GPU
- DataParrallel and OGtransfoxlDataParallel
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
- IMPLEMENT D/LM_TIMEOUT
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
episode length = first 0 mins| based on: openAI used first  mins for 'early game' finetuning, longer for others. Not long anough for long term langauge stuff.

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
EVALUATION_RATE = 1000

VPT_FINETUNE_LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

VPT_WEIGHTS_FILE = 
VPT_MODEL_FILE = 

num_videos = 6000
max_train_steps = num_videos/int(600000/SEQ_LEN)    # 10 mins per video = 600000 ms -> 4687 chunks of 128 frames. want 1000 hours video = 60,000 minutes = 6,000 videos of 10 minutes each
peak_learning_rate = 0.0002
warmup_steps = max_train_steps(0.08)




def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs



### ---------------------- initialise BLC agent
### VPT INIT
agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(VPT_MODEL_FILE)
# To create model with the right environment.
# All basalt environments have the same settings, so any of them works here
env = gym.make("MineRLBasaltFindCave-v0")
agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs).to(DEVICE)
agent.load_weights(VPT_WEIGHTS_FILE)
env.close()
trainable_parameters = agent.policy.parameters()
#initiliase VPT memories to empty
VPT_state = None

### LM INIT
# initialise LM memories to empty
LM_state = None

# define optimizer
optimizer = th.optim.AdamW(
                trainable_parameters,
                lr=peak_learning_rate,
                weight_decay=WEIGHT_DECAY)

lr_schedule = CosineAnnealingWarmupRestarts(optimizer,
                first_cycle_steps=max_train_steps,
                cycle_mult=1.0,
                max_lr=peak_learning_rate,
                min_lr=0.0,
                warmup_steps=1000,
                gamma=1.0)




### ---------------------------- initialise dataset and training
## Data Loader init
data_loader = DataLoader(
    dataset_dir=DATASET+'/train',
    n_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    seq_len = SEQ_LEN,
    n_epochs=EPOCHS)


# start training loop
start_time = time.time()
is_first_frame = th.zeros((BATCH_SIZE, SEQ_LEN), dtype=th.bool).to(DEVICE)
current_video_group_id = 0
current_batch = 0

def BLC_train(data_dir, in_model, in_weights, out_weights):
    global agent, current_video_group_id, current_batch, 

    # get multiple steams of 10 minutes* video across multiple batches. continue until (to ensure lanauge model sees far back langauge)
    for batch_i, (batch_frames, batch_words, batch_actions, video_group_id) in enumerate(data_loader):

        # multiple batches from the same video group will occur in a row. Dont reset memory until a different video group comes along.
        if current_video_group_id != video_group_id:
            is_first_frame[:,...] = True
            agent.policy.LM.reset_mem()
            current_video_group_id = video_group_id


        ### ------------- format input from data loader to agent        
        # format input frames
        x_frames = agent._video_obs_to_agent(batch_frames['img'])
        # format input/label words
        x_words, last_future_words = agent._words_to_agent(batch_words['token_ids'], batch_words['ms'])
        #format action labels
        #actions_formatted = th.zeros([BATCH_SIZE, SEQ_LEN])        # NULL ACTIONS NOT REMOVED: this is probably fine because we still get rid of al lot of null actions (any that happend with no paired word. thjis is done to maintain language integrity and timings). We can mask NULL actions now though. Probably should to maintain comparability to VPT, but this will proabbly hurt performance - VPT suggests getting rid of most but not all NULL actionsm which this probably does. IDK
        action_labels = agent._env_action_to_agent(batch_actions, to_torch=True, check_if_null=False) 
        #actions_formatted[b,t] = action


        ### ----------- feed batch of input sequences (frames and paired language tokens) to agent and get output action and 
        pi_distribution, _, VPT_state, LM_state, LM_loss = agent.policy.get_output_for_observation(
                                                                    ob_words=x_words,
                                                                    ob_frames=x_frames,
                                                                    VPT_state_in=VPT_state,
                                                                    context=is_first_frame,
                                                                    LM_state=LM_state,
                                                                    LM_active_timestep=True,
                                                                    last_future_words=None,
                                                                    last_future_words=last_future_words)

        # (fails with current accumulation). # Make sure we do not try to backprop through sequence in future iterations
        VPT_state = tree_map(lambda x: x.detach(), VPT_state)
        VPT_loss  = -agent.policy.get_logprob_of_action(pi_distribution, action_labels)

        # ------------ optimize model
        BLC_loss = VPT_loss + LM_loss
        BLC_loss.backward()
        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        lr_schedule.step()

        loss_sum += VPT_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0
        if batch_i % EVALUATION_RATE == 0:
            eval_loss = BLC_evaluate()
        
            
        state_dict = agent.policy.state_dict()
        th.save(state_dict, out_weights)


def BLC_evaluate(agent):


    # load examples 128*40 frames
    # load example actions
    # load example actions
    ## Data Loader init
    data_loader = DataLoader(
        dataset_dir=DATASET+'/valid',
        n_workers=1,
        batch_size=2,
        seq_len=SEQ_LEN,
        n_epochs=EPOCHS)

    for batch_i, (batch_frames, batch_words, batch_actions, video_group_id) in enumerate(data_loader):
        video_group_id
        ### ------------- format input from data loader to agent        
        # format input frames
        x_frames = agent._video_obs_to_agent(batch_frames['img'])
        # format input/label words
        x_words, last_future_words = agent._words_to_agent(batch_words['token_ids'], batch_words['ms'])
        #format action labels
        #actions_formatted = th.zeros([BATCH_SIZE, SEQ_LEN])        # NULL ACTIONS NOT REMOVED: this is probably fine because we still get rid of al lot of null actions (any that happend with no paired word. thjis is done to maintain language integrity and timings). We can mask NULL actions now though. Probably should to maintain comparability to VPT, but this will proabbly hurt performance - VPT suggests getting rid of most but not all NULL actionsm which this probably does. IDK
        action_labels = agent._env_action_to_agent(batch_actions, to_torch=True, check_if_null=False) 
        #actions_formatted[b,t] = action

    # agent estimate 10 video sequence batches of 512 with same tgt_len and mem_len






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)





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