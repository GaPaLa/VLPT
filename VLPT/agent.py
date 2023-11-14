import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from lib.action_mapping import CameraHierarchicalMapping, IDMActionMapping
from lib.actions import ActionTransformer
from lib.VLPT_policy import MinecraftAgentPolicy
from lib.torch_util import default_device_type, set_default_torch_device

from transformers import TransfoXLTokenizer

# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def validate_env(env):
    """Check that the MineRL environment is setup correctly, and raise if not"""
    for key, value in ENV_KWARGS.items():
        if key == "frameskip":
            continue
        if getattr(env.task, key) != value:
            raise ValueError(f"MineRL environment setting {key} does not match {value}")
    action_names = set(env.action_space.spaces.keys())
    if action_names != set(TARGET_ACTION_SPACE.keys()):
        raise ValueError(f"MineRL action space does match. Expected actions {set(TARGET_ACTION_SPACE.keys())}")

    for ac_space_name, ac_space_space in TARGET_ACTION_SPACE.items():
        if env.action_space.spaces[ac_space_name] != ac_space_space:
            raise ValueError(f"MineRL action space setting {ac_space_name} does not match {ac_space_space}")


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR @double check this: the paper shows integer scaling
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img
def resize_video(vid, target_resolution):

    # For your sanity, do not resize with any function than INTER_LINEAR
    vid_resized = np.zeros([vid.shape[0],target_resolution[0],target_resolution[1]],3)

    for f in range(vid.shape[0]):
        vid_resized[f] = cv2.resize(vid[f], target_resolution, interpolation=cv2.INTER_LINEAR)

    return vid_resized


class MineRLAgent:
    def __init__(self, env=None, device=None, policy_kwargs=None, pi_head_kwargs=None, LM_type=None, LM_TIMEOUT_RATE=4, L_SEQ_LEN=256, dtype=th.float32, LM_ONLY=False):
        if env and not env==-999:
            validate_env(env)
        self.LM_type = LM_type

        ## args
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        self.IDM_action_mapper = IDMActionMapping(n_camera_bins=11)

        
        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS
        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space, device=device)
        self.policy = MinecraftAgentPolicy(**agent_kwargs, LM_type=LM_type, LM_TIMEOUT_RATE=LM_TIMEOUT_RATE, L_SEQ_LEN=L_SEQ_LEN, dtype=dtype, LM_ONLY=LM_ONLY).to(device)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

        # LM
        if LM_type != None:
            self.tokenizer = TransfoXLTokenizer.from_pretrained(LM_type)
            self.BOS = 2
            self.SILENCE_TOKEN=2
            self.LM_word_context = th.full([1,1],self.BOS, dtype=th.long).to(self.device)
            self.LM_hidden_state = None
            self.LM_TIMEOUT=LM_TIMEOUT_RATE
            self.LM_starter_words = th.full([0,0],self.BOS, dtype=th.long).to(self.device)
            self.previous_LM_output = None

        # hidden state management
        self.VPT_hidden_state = self.policy.initial_state(1)
        self.Xattn1_mems = th.zeros([1,128,2048], dtype=th.float32).to(self.device) 
        self.Xattn2_mems = th.zeros([1,128,1024], dtype=th.float32).to(self.device) 

        self.current_timestep = 0
        

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()


    def reset(self):
        self.current_timestep=0

        """Reset agent to initial state (i.e., reset hidden state) during inference. Does not work with batch_size>1"""
        # reset VPT mems
        self.VPT_hidden_state = self.policy.initial_state(1)
        #reset LM mems
        if self.LM_type != None:
                self.LM_word_context = th.full([1,1],self.BOS, dtype=th.bool).to(self.device)
        # reset Xattn mems
        self.Xattn1_mems = th.zeros([1,128,2048], dtype=th.float32).to(self.device) 
        self.Xattn2_mems = th.zeros([1,128,1024], dtype=th.float32).to(self.device) 

        
    def _env_obs_to_agent(self, obs_frames):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(obs_frames["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        
        return agent_input

    def _video_obs_to_agent(self, video_frames, ms):
        imgs = [resize_image(frame, AGENT_RESOLUTION) for frame in video_frames]
        # Add time and batch dim
        imgs = np.stack(imgs)[None]
        video_obs = {"img": th.from_numpy(imgs).to(self.device)}
        video_obs['ms'] = ms
        return video_obs

    def _words_to_agent(self, obs_words, obs_frames):
        words = obs_words['words']
        words_ms = obs_words['ms']
        F_SEQ_LEN = obs_frames['img'].shape[1]
        L_SEQ_LEN = F_SEQ_LEN//self.LM_TIMEOUT
        BATCH_SIZE = len(words)

        ## take in list of list of every word in episode, and list of list of when those words occur in time (in ms). 

        ## PAIRS WORDS WITH FRAMES AND CREATES EQUAL SHAPED WORD TENSOR TO FRAM TENSOR.
        ## NOTE: TO MAINTIN CAUSALITY: word 0 is generated by frame 0. When generating word 0, we therefore cannot ahave access to word 0. we must have acces to frame 0 and words -1...
        # At frame T, LM must estimate the next word T given previous frames T-1... and previous words T-1...
        # at timestep T therfore, the input to LM is frame2 T..., words T-1...
        # therefore, every frame is paired with the word before (every word is paired with the )


        num_frames = obs_frames['img'].shape[1]

        ## convert word input to tensor
        # for every batch, get sentence, tokenize, pad, format ms

        ### Format ob_words so that every frame has an associated word, using silence token insertion when there are no words for a frame.
        ### ob_words may have any sequence length, may be word tokens for entire sequence.
        ## save results as variable langauge_tokens
        temporal_language_tokens = th.full([BATCH_SIZE, (L_SEQ_LEN*self.LM_TIMEOUT)+1], self.SILENCE_TOKEN, dtype=th.long)    # need to get one more langauge token than frames at end so that we have a word to predict i.e. this tensor is not just input but also hold labels
        # NOTE: BOS must be in given tokens and it must occur ata  negative timestamp to indicate that the first frame is allowed to be apired wth that token and passed to LM as one input from past observations
        for b in range(BATCH_SIZE):
            words_ms_tensor_1traj = np.asarray(words_ms[b], dtype=np.uint32)
            tokens_queue=[]
            for t in range(L_SEQ_LEN): #### D MUST BE DIVISIBLE BY NUMBER OF FRAMES
                
                #check for langauge tokens that occur during the 50ms span of each frame
                word_index = np.where(  (words_ms_tensor_1traj >= obs_frames['ms'][b,t] - 50*self.LM_TIMEOUT)
                                       &(words_ms_tensor_1traj <  obs_frames['ms'][b,t]))
                # work out token_ids of occured words and append to buffer of tokens to be assigned to a frame
                for i in range(word_index[0].shape[0]):
                    token_id = words[b][word_index[0][i]]
                    print('seen token', token_id,'@ms:',words_ms_tensor_1traj[word_index[0][i]])

                    tokens_queue.append(token_id)
    
                # if language tokens available in buffer, associate oldest wth current frame and remove from buffer.
                if len(tokens_queue)>0: # if there were skipped over tokens during LM timeout (D), they shuold be added to the queue and can be popped one at a time now during the silence
                    temporal_language_tokens[b,t*self.LM_TIMEOUT] = tokens_queue.pop(0)

            if not len(tokens_queue)==0:
                print( "DATAWARNING: while assigning words to frames, ",str(len(tokens_queue))," tokens were unassigned due to more words than frames given D" )


        input_words = temporal_language_tokens[:,:-1] # input words are all words except last one
        labels = temporal_language_tokens[:,1:] # label words are the tokens 1 ahead of input tokens


        return input_words.to(self.device), labels.to(self.device)

    def _agent_words_to_string(self, words):
        strings = self.tokenizer.decode(words)    
        return strings

    def _agent_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed


    def _IDM_action_to_env(self, agent_action):
        """Turn output from policy into action for MineRL"""
        # This is quite important step (for some reason).
        # For the sake of your sanity, remember to do this step (manual conversion to numpy)
        # before proceeding. Otherwise, your agent might be a little derp.
        action = {
            "buttons": agent_action["buttons"],
            "camera": agent_action["camera"]
        }
        minerl_action = self.IDM_action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed








    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action

















    def add_inference_starter_words(self, starter_words):
        if starter_words:
            starter_words = self.tokenizer(starter_words)['input_ids']
            self.LM_starter_words = th.tensor(starter_words, dtype=th.long).reshape([1,len(starter_words)]).to(self.device)


    # RL inference
    # can pass text to start the model off with as plaintext e.g. 'Hi guys today I'm going to build a house'
    def get_action(self, obs_frames):

        # allow for passing starter word context, but the model generates its own tokens and feeds them back into the context as it runs
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally (within this class). To reset it,
        call `reset()`.
        """
        if self.current_timestep%self.LM_TIMEOUT==0:
            #use starter words until done
            if self.current_timestep < self.LM_starter_words.shape[1]*self.LM_TIMEOUT:
                current_word = self.LM_starter_words[0,self.current_timestep//self.LM_TIMEOUT].view([1,1]).to(self.device)
                #print('CONTEXT:', self.tokenizer.decode(current_word.reshape([1]) ))
            # use last word outputted from LM as input to LM
            else:
                current_word = self.LM_word_context[0,-1].view([1,1]).to(self.device)
                #print('AUTOREG: ', self.tokenizer.decode(current_word.reshape([1]) ))
                
        # placeholder if LM inactive timestep
        else:
            current_word = th.full([1,1], self.BOS, dtype=th.long).to(self.device)

        obs_frames = self._env_obs_to_agent(obs_frames)

        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, agent_action_pd, self.VPT_hidden_state, self.LM_hidden_state, self.previous_LM_output, vpred_word, self.Xattn1_mems, self.Xattn2_mems = self.policy.act(
                                                                                                            obs_frames=obs_frames,
                                                                                                            current_timestep=self.current_timestep,
                                                                                                            first=self._dummy_first, 
                                                                                                            VPT_state=self.VPT_hidden_state,
                                                                                                            LM_state=self.LM_hidden_state,
                                                                                                            obs_word=current_word,
                                                                                                            stochastic=True, ####### @try deterministc?
                                                                                                            previous_LM_output=self.previous_LM_output,
                                                                                                            Xattn2_state = self.Xattn2_mems,
                                                                                                            Xattn1_state = self.Xattn1_mems
                                                                                                            )
        minerl_action = self._agent_action_to_env(agent_action)

        # print either estimated word or current word context word
        if self.current_timestep%self.LM_TIMEOUT==0:

            if self.current_timestep < self.LM_starter_words.shape[1]*self.LM_TIMEOUT:
                print('CONTEXT:', self.tokenizer.decode(current_word.reshape([1]) ))
            # use last word outputted from LM as input to LM
            if self.current_timestep >= self.LM_starter_words.shape[1]*self.LM_TIMEOUT -self.LM_TIMEOUT:
                print('PREDICTED: ',self.tokenizer.decode(vpred_word.reshape([1])))

        # if active LM timestep and we are past the starter word context, append LM output back into word context
        if self.current_timestep%self.LM_TIMEOUT==0:

            NO_AUTOREGRESSIVE = True
            if NO_AUTOREGRESSIVE:
                self.LM_word_context = th.cat([self.LM_word_context, th.full([1,1],2,).to(self.device)], dim=1).to(self.device)
            else:
                if self.current_timestep >= self.LM_starter_words.shape[1]*self.LM_TIMEOUT -self.LM_TIMEOUT:
                    self.LM_word_context = th.cat([self.LM_word_context, vpred_word], dim=1).to(self.device)


        self.current_timestep += 1
        return minerl_action, self.LM_word_context[0,-1]