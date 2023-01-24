import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from lib.action_mapping import CameraHierarchicalMapping
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


class MineRLAgent:
    def __init__(self, env, device=None, policy_kwargs=None, pi_head_kwargs=None):
        validate_env(env)

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
        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS
        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space, device=device)
        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

        # LM
        self.tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")
        self.BOS = 2

        # hidden state management
        self.VPT_hidden_state = self.policy.initial_state(1)
        self.LM_hidden_state = None
        self.LM_word_context = th.full([1,1],self.BOS, dtype=th.bool)
        self.current_timestep=0

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state) during inference. Does not work with batch_size>1"""
        self.VPT_hidden_state = self.policy.initial_state(1)
        self.LM_word_context = th.full([1,1],self.BOS, dtype=th.bool)
        self.current_timestep=0

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        
        return agent_input

    def _video_obs_to_agent(self, video_frames, ms):
        imgs = [resize_image(frame, AGENT_RESOLUTION) for frame in video_frames]
        # Add time and batch dim
        imgs = np.stack(imgs)[None]
        video_obs = {"img": th.from_numpy(imgs).to(self.device)}
        video_obs['ms'] = ms
        return video_obs

    def _words_to_agent(self, words, word_ms, ob_frames):
        ## take in list of list of every word in episode, and list of list of when those words occur in time (in ms). 

        ## PAIRS WORDS WITH FRAMES AND CREATES EQUAL SHAPED WORD TENSOR TO FRAM TENSOR.
        ## NOTE: TO MAINTIN CAUSALITY: word 0 is generated by frame 0. When generating word 0, we therefore cannot ahave access to word 0. we must have acces to frame 0 and words -1...
        # At frame T, LM must estimate the next word T given previous frames T-1... and previous words T-1...
        # at timestep T therfore, the input to LM is frame2 T..., words T-1...
        # therefore, every frame is paired with the word before (every word is paired with the )


        LM_TIMEOUT = 2
        FRAMERATE = 20
        SILENCE_TOKEN_ID = 2 # 2 in transfo_xl is ','. performs second best to ' ', but transfo_xl has no ' ' token since it is word-level tokenizer. #1437 in OPT is space token. best performing in OPT accoding to rough experiments on gutenber performance with different random tokns inserted at the same positions. # IF TRANSCRIBER GIVE NO COMMA, USE COMMA. comma=6. newline character=50118. single space=1437 luckily, in this tokenizer, stray spaces are classified as separate characters isntead of extensions of real word parts, so it is an intuitive 1:1 translation of extra spaces from human langauge to model. Hopefully it is therefore much easier to learn! # TEST: INSERT TOKENS AS NOISE TO NORMAL LANGUAGE INPUT AND SEE WHICH IS LEAST DESTRUCTIVE TO PERFORMANCE
        num_frames = ob_frames['img'].shape[1]

        ## convert word input to tensor
        batch_size = len(words) # words is a list of strings, each string is the language input over a section of video. The words must line up with the video frames being fed into the model
        max_len = 0
        for b in range(len(words)):
            if len(words[b])> max_len:
                max_len = len(words[b])
        token_ids_tensor = th.zeros([batch_size, max_len], dtype=th.long).to(self.device)
        ms_tensor = th.full([batch_size, max_len], float('inf'), dtype=th.uint32).to(self.device) # set to inf so that padded words never match with frame occurences. @limit of uint32 size limits max length an episode can be but complicated to fix and it limits it to a huge number anyway
        # for every batch, get sentence, tokenize, pad, format ms
        # although 
        for b in range(len(words)):
            token_ids = self.tokenizer(words[b])["token_ids"] # IF WORD INPUT IS EMPTY, NO FRAMES CAN PASS THROUGH LM. ALWAYS PASS AT LEAST BOS TOKEN SO IT ALWAYS ALLOWS INFORMATION EVEN WITHOUT LANGUAGE. FOR OPT/GPT2/GPT3 THIS IS TOKEN_ID=2. THIS IS ALREADY DONE IN THIS TOKENIZER.
            token_ids_tensor[b,0:len(token_ids)] = token_ids
            ms_tensor[b,0:len(token_ids)] = word_ms[b]
        ob_words = { "token_ids":token_ids_tensor,
                    "ms":ms_tensor }


        ### Format ob_words so that every frame has an associated word, using silence token insertion when there are no words for a frame.
        ### ob_words may have any sequence length, may be word tokens for entire sequence.
        ## save results as variable langauge_tokens
        language_tokens = th.zeros([batch_size, num_frames//LM_TIMEOUT], dtype=th.long)
        # NOTE: BOS must be in given tokens and it must occur ata  negative timestamp to indicate that the first frame is allowed to be apired wth that token and passed to LM as one input from past observations
        for b in batch_size:
            
            tokens_queue=[]
            for t in num_frames//LM_TIMEOUT: #### D MUST BE DIVISIBLE BY NUMBER OF FRAMES
                
                #check for langauge tokens that occur during the 50ms span of each frame
                word_index = th.where(  (ob_words['ms'][b] >= ob_frames['ms'][b,t*LM_TIMEOUT] - 50*LM_TIMEOUT)
                                       &(ob_words['ms'][b] <  ob_frames['ms'][b,t*LM_TIMEOUT]))
                
                # work out token_ids of occured words and append to buffer of tokens to be assigned to a frame
                for i in range([0]*word_index[0].shape[0]):
                    token_id = ob_words[b,word_index[0][i]]
                    tokens_queue.append(token_id)
    
                # if language tokens available in buffer, associate oldest wth current frame and remove from buffer.
                if len(tokens_queue)>0: # if there were skipped over tokens during LM timeout (D), they shuold be added to the queue and can be popped one at a time now during the silence
                    language_tokens[b,t] = tokens_queue.pop(0)
                # if no langauge tokens available, insert silence token
                else:
                    language_tokens[b,t] = SILENCE_TOKEN_ID # if no words, use silence token.
            assert len(tokens_queue)==0, "while assigning words to frames, "+len(tokens_queue)+" tokens were unassigned due to not enough silence at the end"

        return language_tokens

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

    # RL inference
    # can pass text to start the model off with as plaintext e.g. 'Hi guys today I'm going to build a house'
    def get_action(self, minerl_obs, starter_words=None):

        # allow for passing starter word context, but the model generates its own tokens and feeds them back into the context as it runs
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally (within this class). To reset it,
        call `reset()`.
        """



        if starter_words:
            starter_words = self.tokenizer(starter_words)['input_ids']
            self.word_context = th.tensor(starter_words).reshape([1,len(starter_words)], dtype=th.bool)



        current_frame = self._env_obs_to_agent(minerl_obs)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.VPT_hidden_state, _, self.LM_hidden_state, self.LM_word_context = self.policy.act(
                                                                                                            current_frame, 
                                                                                                            self._dummy_first, 
                                                                                                            self.VPT_hidden_state,
                                                                                                            self.LM_hidden_state,
                                                                                                            self.LM_word_context,
                                                                                                            self.current_timestep,
                                                                                                            stochastic=True ####### @try deterministc?
                                                                                                            )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action
