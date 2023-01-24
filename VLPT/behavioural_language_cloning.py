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
- Figure out how VPT, LM put sequences across multiple batches: how data physically laid, out, how hidden state reset.
- Figure out good max_len to use
- Figure out when to backprop
- Fix gradient accumulation
- First vs mem_reset()
- Do: audio -> tokens + tokens ms -> save to file
- Edit DataLoader: load transcipt.file
- add val_loss iteration every 100 batches: VLPT_val_loss, LM_val_loss, LM_wt103_loss
- save model every 10000 batches: for MineRL assessment through time.
- Data Loader: (if doesnt fit in batch,m just discard batch, end episode early) (send episode_ended signal to clear hidden state when done. How to clear hidden state? how hidden staet batch???)


--- OPTIMIZE
- put VPT and LM on different GPU
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
- currently, TransformerXL's softmx classification layer only either support gettings the loss or predicting wor dprobabilities, not both. This means that it must be passed through twice in ordre to get both, which is slow. TO optimize, finda  way to sample AdaptiveLogSoftmax while getting loss.


- check PaLI/Flamingo paper for training procedure: KL, losses...


--- WORDS
- code YT_link -> audio collector
- code audio -> transcript + timestamps
- create script to take audio from video, check if WPM is good (180 - 240), then save audio somewhere safe if good, along with video link to a file.

--- VIDEO
- manually label 8000 frames spam/ham
- train SVM+frozen CLIP as spamham
- create script to select a video, select a few seconds randomly, download whole video if 80% gathered frames are clean

--- INITIAL TESTING
- Finetune pure VPT on collected video dataset using VLPT codebase with langauge model set as not active.
- on collected video audio data with NULLS filtered out, check how often silence tokens appear (some balance between peak and average) and set D appropriately
- Finetune Transfo_XL on collected words dataset
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

from lib.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax

if th.cuda.is_available():
    device = th.device('cuda')
    print("USING CUDA")
else:
    device = th.device('cpu')
    print("USING CPU")




EPOCHS = 20
# Needs to be <= number of videos
BATCH_SIZE = 8
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 12
DEVICE = "cuda"

LOSS_REPORT_RATE = 100

LEARNING_RATE = 0.000181
WEIGHT_DECAY = 0.039428
MAX_GRAD_NORM = 5.0

def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights):



    ### VPT INIT
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    env = gym.make("MineRLBasaltFindCave-v0")
    agent = MineRLAgent(env, device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    agent.load_weights(in_weights)
    env.close()
    policy = agent.policy
    trainable_parameters = policy.parameters()
    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    ### LM INIT
    # ALTER PYTHON FILES WITH UPTADTED EFFICIENT TRANSFORMER XL SOFTMAX FUNCTION
    

    ## Data Loader init
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)










    loss_sum = 0
    for batch_i, (batch_frames, batch_words, batch_actions, batch_episode_id) in enumerate(data_loader):
        batch_size = batch_frames['img'].shape[0]
        sequence_length = batch_frames['img'].shape[1]

        actions_formatted = th.zeros([batch_size, sequence_length])
        for b in range(batch_size):
            for t in range():
                # NULL ACTIONS NOT REMOVED: this is probably fine because we still get rid of al lot of null actions (any that happend with no paired word. thjis is done to maintain language integrity and timings). We can mask NULL actions now though. Probably should to maintain comparability to VPT, but this will proabbly hurt performance - VPT suggests getting rid of most but not all NULL actionsm which this probably does. IDK
                action = agent._env_action_to_agent(batch_actions, to_torch=True, check_if_null=False) 
                actions_formatted[b,t] = action

        frames_formatted = th.zeros(len(batch_frames['img']), 128,128,3)
        frames_formatted = agent._video_obs_to_agent(batch_frames['img'])
        
        words_formatted = agent._words_to_agent(batch_words['ob_words'], batch_words['ms'])
        

        if episode_id_combo not in episode_hidden_states:
            # TODO need to clean up this hidden state after worker is done with the work item.
            #      Leaks memory, but not tooooo much at these scales (will be a problem later).
            episode_hidden_states[episode_id] = policy.initial_state(1)
        agent_state = episode_hidden_states[episode_id]

        pi_distribution, v_prediction, VPT_state, LM_state, LM_loss = policy.get_output_for_observation(
            frames_formatted,
            words_formatted,
            agent_state,
            dummy_first
        )

        VPT_loss  = -policy.get_logprob_of_action(pi_distribution, actions_formatted)

        pred_hid = LM_output.hidden_states[:-tgt_len:]
        LM_labels = ### construct labels              https://github.com/huggingface/transformers/blob/main/src/transformers/models/transfo_xl/modeling_transfo_xl.py bringing code from inside the model outside  to customise its utility. internally, num albels = num input tokens and so the loss is not calculated for each output: inefficient. so the loss calculation is rbought out from within the model and optimised 
        LM_labels = LM_in_words.roll(-1, dims=1)
        if last_future_words:
        LM_labels[:,-1] = last_future_words # if we have future words to make labels with use thsi, otherwise still use broken labels as labels, loss will obviously not be applicable but presumably we ommitted future_word knowing this and do not intend to use the loss, 
        softmax_output = self.crit(pred_hid, labels)
        prediction_scores = softmax_output.view(bsz, tgt_len, -1) if labels is None else ()

        if labels is not None:
            losses = softmax_output.view(bsz, tgt_len - 1)
            # Avoids from incorporating padding (-100) tokens into loss value
            loss = losses[losses != 0].mean()
        else:
            losses, loss = None, None

        LM_loss = policy.net.LM.crit(pred_hid, labels)



        # Make sure we do not try to backprop through sequence in future iterations
        # (fails with current accumulation).
        # @ implementing backprop through sequence/time. 
        # Batch accumulation method removed for standard batching
        VPT_state = tree_map(lambda x: x.detach(), VPT_state)
        episode_hidden_states[episode_id] = VPT_state
        

        # Finally, update the agent to increase the probability of the
        # taken action.
        # Remember to take mean over batch losses
        # In old VPT, if input action was NULL, remove from training data. However, this disrupts the word/frame timing for the LM,
        # so we need to let the model predict from these actions but not train it to predict them.
        # the VPT paper also said that removing triplets of actions is best for performance but they didnt do this because they only realised later.
        # In order to make results more comparable to VPT, I do not do this.
        batch_loss = (-log_prob / BATCH_SIZE)
        batch_loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0
        
        # delete the hidden states for finished episodes

state_dict = policy.state_dict()
th.save(state_dict, out_weights)


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