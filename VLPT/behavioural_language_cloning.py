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






import os
!conda install python=3.8
os.system("sudo apt install git")
os.system("sudo add-apt-repository ppa:openjdk-r/ppa && sudo apt-get update && sudo apt-get install openjdk-8-jdk")
!pip install git+https://github.com/minerllabs/minerl@v1.0.0
!pip install -r "requirements.txt"
#language prerequisites 
!conda install cudatoolkit=11.3
!pip install transformers






if th.cuda.is_available():
    device = th.device('cuda')
    print("USING CUDA")
else:
    device = th.device('cpu')
    print("USING CPU")





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
    for batch_i, (batch_images, batch_actions, batch_words, batch_episode_id) in enumerate(data_loader):
        batch_loss = 0
        for trajectory_frames, trajectory_words, pre_trajectory_word_context, trajectory_actions, episode_id in zip(batch_images, batch_actions, batch_words, batch_episode_id):
            
            # get frame index
            LM_state = policy.give_context_words(pre_trajectory_word_context)

            for frame, word, action, episode_id in zip(trajectory_frames, trajectory_words, trajectory_actions, episode_id):
                action_formatted = agent._env_action_to_agent(action, to_torch=True, check_if_null=True)
                if action_formatted is None:
                    # Action was null
                    continue

                frames_formatted = torch.zeros(len(trajectory_frames), 128,128,3)
                for i in range(trajectory_frames):
                    frames_formatted[i,:,:,:] = agent._env_obs_to_agent({"pov": trajectory_frames[i]})
                
                

                #@ is this even necesary for transformer? pretty sure for residualrecurrentblock, state_out = state_in, therefore state_in is always initial state
                if episode_id not in episode_hidden_states:
                    # TODO need to clean up this hidden state after worker is done with the work item.
                    #      Leaks memory, but not tooooo much at these scales (will be a problem later).
                    episode_hidden_states[episode_id] = policy.initial_state(1)
                agent_state = episode_hidden_states[episode_id]

                pi_distribution, v_prediction, new_agent_state = policy.get_output_for_observation(
                    frames_formatted,
                    words_formatted,
                    agent_state,
                    dummy_first
                )

                log_prob  = policy.get_logprob_of_action(pi_distribution, action_formatted)

                # Make sure we do not try to backprop through sequence
                # (fails with current accumulation).
                # @ implementing backprop through sequence/time. 
                # Batch accumulation method removed for standard batching
                new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
                episode_hidden_states[episode_id] = new_agent_state
                
                # Finally, update the agent to increase the probability of the
                # taken action.
                # Remember to take mean over batch losses
                loss = -log_prob / BATCH_SIZE
                batch_loss += loss.item()
                loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            loss_sum = 0

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