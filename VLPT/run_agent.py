from argparse import ArgumentParser
import pickle, os
import torch as th

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

from agent import MineRLAgent, ENV_KWARGS

def main(model, weights):
    env = HumanSurvival(**ENV_KWARGS).make()
    print("---Loading model---")
    print(os.getcwd())
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device=th.device('cpu'))
    if weights:
        agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()

    while True:
        minerl_action, pred_word = agent.get_action(obs, starter_words='Hi guys, today in my Minecraft world we are going to')
        obs, reward, done, info = env.step(minerl_action)

        print(agent.tokenizer.decode(pred_word))
        env.render()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=False, help="Path to the '.weights' file to be loaded.", default=None)
    parser.add_argument("--model", type=str, required=False, help="Path to the '.model' file to be loaded.", default='/media/idmi/DISSERTATN/VLPT/VLPT/2x.model' )

    args = parser.parse_args()

    main(args.model, args.weights)
