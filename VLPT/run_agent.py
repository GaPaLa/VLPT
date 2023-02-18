from argparse import ArgumentParser
import pickle, os
import torch as th

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS


model = '/media/idmi/DISSERTATN/VLPT/VLPT/2x.model'
weights = 'none'

env = HumanSurvival(**ENV_KWARGS).make()
print("---Loading model---")
print(os.getcwd())
agent_parameters = pickle.load(open(model, "rb"))
policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device=th.device('cuda'))
if weights!='none':
    agent.load_weights(weights)

print("---Launching MineRL enviroment (be patient)---")
obs = env.reset()
agent.add_inference_starter_words('Hi guys, welcome to my new Minecraft world! Today we are going to start off by')

while True:
    minerl_action, pred_word = agent.get_action(obs_frames=obs)
    obs, reward, done, info = env.step(minerl_action)

    #print(agent.tokenizer.decode(pred_word))
    env.render()