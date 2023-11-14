from argparse import ArgumentParser
import pickle, os
import torch as th

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS


DEVICE='cuda'

model = '/media/idmi/DISSERTATN/VLPT/VLPT/2x.model'
VPT_WEIGHTS_FILE = '/home/idmi/Downloads/VLPT_1300_.weights'





env = HumanSurvival(**ENV_KWARGS).make()
print("---Loading model---")
agent_parameters = pickle.load(open(model, "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, LM_type="transfo-xl-wt103", LM_TIMEOUT_RATE=4, L_SEQ_LEN=128, dtype=th.bfloat16, device=th.device(DEVICE))
agent.load_weights(VPT_WEIGHTS_FILE)


### ---------------------- initialise BLC agent

print("---Launching MineRL enviroment (be patient)---")
obs = env.reset()
agent.add_inference_starter_words("Hi guys, welcome to my new Minecraft world! Today we are going to just be exploring. we're going to just run and sprint and walk around ")

while True:
    minerl_action, pred_word = agent.get_action(obs_frames=obs)
    obs, reward, done, info = env.step(minerl_action)

    #print(agent.tokenizer.decode(pred_word))
    env.render()