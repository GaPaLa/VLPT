from transformers import OPTForCausalLM

from copy import deepcopy
from email import policy
from typing import Dict, Optional

import numpy as np
import torch as th
from gym3.types import DictType
from torch import nn
from torch.nn import functional as F

from lib.action_head import make_action_head
from lib.action_mapping import CameraHierarchicalMapping
from lib.impala_cnn import ImpalaCNN
from lib.normalize_ewma import NormalizeEwma
from lib.scaled_mse_head import ScaledMSEHead
from lib.tree_util import tree_map
from lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from lib.misc import transpose
from lib.masked_gated_cross_attention import MaskedGatedCrossAttention
from lib.mlp import MLP

















class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(th.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(th.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img.to(dtype=th.float32)
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))
















class MinecraftPolicy(nn.Module):
    """

        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        device,
        recurrence_type="transformer", 
        impala_width=8,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=2048,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        img_shape=[128,128,3],
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={'post_pool_groups': 1},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=16,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=4,
        recurrence_is_residual=True,
        timesteps=128,
        use_pre_lstm_ln=False,  # Not needed for transformer
        **unused_kwargs,
    ):
        super().__init__()
        self.device = device

        assert recurrence_type == "transformer"
        self.recurrence_type = "transformer"
        active_reward_monitors = active_reward_monitors or {}
        self.single_output = single_output
        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )
        
        # Define Language Model
        self.LM = OPTForCausalLM.from_pretrained("facebook/opt-350m") #, @later, load from stored weights
        
        # define cross atention layers from VPT->LM, LM->VPT
        self.Xattn_VPT_LM = MaskedGatedCrossAttention(
            embed_dim=self.LM.model.config.hidden_size, # output LM-size output to LM
            kvdim=hidsize, # embedding dimensions for VPT transformer layers
            num_heads=attention_heads,
            ffw_dim=hidsize*pointwise_ratio, # FFW hidden size should be same architecturally as FFW in LM (i.e. ratio between transformer token size and FFW hidden layer should be same) (x2 to account for multimodal tokens)
            batch_first=True,
            dtype=th.float32
        )
        self.Xattn_LM_VPT = MaskedGatedCrossAttention(
            embed_dim=hidsize, # output VPT-size output to VPT 
            kvdim=self.LM.model.config.hidden_size,
            num_heads=attention_heads,
            ffw_dim=hidsize*pointwise_ratio, # use same as FFW (x2 to account for multimodal tokens)
            batch_first=True,
            dtype=th.float32
        )

        # VPT transformer layers
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers)
        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        return self.hidsize






    # --------------       ---------------         -------------       -----------       KEY ARCHITECTURE DEFINITION
    ## NOTE OF HOW FORWARD WORKS AT END OF THIS CLASS:
    def forward(self, ob_words, ob_frames, VPT_state_in, context, inference=False, LM_state_in=None, past_xattn=None, LM_active=True):

        ## useful stuff for later
        num_frames = ob_frames['img'].shape[1]
        num_words = ob_words['token_ids'].shape[1]
        batch_size = ob_frames['img'].shape[0]
        print("words, frmes="+str(num_words)+" "+str(num_frames))
        assert num_frames==num_words, "The frames and words tensors must match"
        first = context["first"]
        LM_TIMEOUT = 4 # referred to also as 'D'. LM gets input from VPT and gives output to VPT every D timesteps to reduce silence token noise.
        


        # --------------------------- START PASSING THOURGH VPT MODEL
        
        ### pass input frames through CNN section
        if inference: # if doing inference, since VPT has VPT_state to keep context, only process newest frame
            x = self.img_preprocess(ob_frames["img"][:,-1])
        else:
            x = self.img_preprocess(ob_frames["img"])

        x = self.img_process(x)

        ### pass processed frames through VPT Agent transformer layer #1
        x, VPT1_state_out = self.recurrent_layer(x, first, [VPT_state_in[0]], start_block=0, end_block=1)
        print('VPT1 out shape',x.shape)


        # ----------------------------------------- INSERT LM
        
        ### copy VPT_transofrmer_layer1 output to pass the LM via cross-attention, leaving original output as residual around LM (as in Flamingo-style gated cross-attention. see: masked_gated_cross_atention.py)
        # implement LM timeout - D
        LM_frames = x.clone()[:,::LM_TIMEOUT,:] #get every Dth frame. means that LM only sees as many framse as words i.e. frames are as spares as words. We CAN train on all past frames every langauge token, however, this results in 2048*D frames cross-attending with language token #2048 as opposed to just 2048 frames corss attending with langauge token #2048. Maybe that's okay? or preferable? It sounds expensive, ill do the sparse frames method. plus, this simplifies the X_attn mask #@r2
        LM_in_words = ob_words['token_ids'][:,::LM_TIMEOUT] # crop word equally. This should have been modified by agent.py . words_to_agent() to only be silent tokens
        

        ### convert input word token ids to embeddings. since LM(words) works from token_indices and we need to pass
        # VPT representations as raw vectors, we need to use LM(words, from_embeddings=True),
        # which requires us to pre-embed to token ids
        word_embeddings = self.LM.model.decoder.embed_tokens(LM_in_words)
        word_embeddings = self.LM.model.decoder.project_in(word_embeddings) # for more model expressiveness, edited modelling_opt.py so that we can input full LM embeddings (1024) rather then word representations (512), i.e. work just before positional embeddings are added, which are also 1024 big.
        
        ### Gated cross attention: FUSE output from VPT transformer layer 1 and language input tokens
        # need CAUSAL ATTENTION MASK in cross attention, so langauge tokens cant cross attend to future frames, since the inputted frames and words tensors both have future and past information relative to each other
        # frames and words are ordered such that at frame index 0, word index 0 has already occured. therefore we can take both as input to estimate the next word
        # word 0 can see frame 0 (because frame 0 occurs before the word does).
        Xattn_mask=th.ones([LM_in_words.shape[1], LM_frames.shape[1]]) # create causal mask between language tokens and frames. every frame is paired with the previous word emitted, so that at the current frame we predic the next word from the current frame (and previous frames witha ttentions) and rpevious words. For more details look at agent.py words_to_agent()
        for q in range(LM_in_words.shape[1]):
            Xattn_mask[q, 0:q+1] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.
        fused = self.Xattn_VPT_LM(x=LM_frames, y=word_embeddings, attn_mask=Xattn_mask)
        print('xattn1 out shape',fused.shape)
        #### Feed LM fused input tokens & predict raw LM output
        if inference: # if doing inference and not reached 2048 tokens context, just predict next use cached keys. @FIX: NEED TO DO TOKEN-RECOMPUTATION IF CONTEXT>2048 (i.e. more than D*2048 frames have occurred)
            if LM_active:
                if past_xattn.shape[1]<2047:
                    fused, LM_state_out = self.LM.model.forward(inputs_embeds=fused[:,-1,:], project_embeds=False, past_keys=LM_state_in, use_cache=True)['last_hidden_state'] # IF DOING INFEREMCE AND CONTEXT LIMIT NOT REACHED, just predict from current token using past keys.
                else: # we CAN compute with 2048 but if its easier to assume its already been cropped to 2047 (so we can add current token to that and make 2048) and BOS moved by act() function after it reached high, so we always recompute at this length, besides, onlt slows down performance for one frame where the  re-computing method has to take over from the state-passing method.
                    past_xattn = th.cat([past_xattn,fused[:,-1,:].clone().detach()], axis=1) # tack on history of tokens to be re-computed. now input is 2048 tokens
                    fused, LM_state_out = self.LM.model.forward(inputs_embeds=past_xattn, project_embeds=False, use_cache=True)['last_hidden_state'] # since future langauge tokens are masked in x-attention, we don't have to mask them here. IF WE DO, LM USES OPPOSITE ASKING SCHEME TO X-ATTENTION. USE LM_mask = (language_maks=0)
            else:
                fused=past_xattn[:,-1,:] # if at a timestep where LM is timeout, just copy the last output from LM to VPT
        else:
            fused = self.LM.model.forward(inputs_embeds=fused, project_embeds=False)['last_hidden_state']
        print('LM out shape',fused.shape)
        #so that there is an LM output for every frame despite D LM timeout, repeat every LM output in time D times. 
        fused = th.repeat_interleave(fused, LM_TIMEOUT, dim=1)
        fused = fused[:,:num_frames,:].reshape([batch_size,num_words,1024]) #Make sure not to output too many values if the input frames is smaller than D
        print('LM repeated out shape',fused.shape)

        #### From raw LM output, predict word classes (for language modelling loss)
        LM_words = fused.clone() # we need separate LM loss and RL loss - 'l' branch is used for RL, 'LM_words' it used for LM 
        LM_words = self.LM.model.decoder.project_out(LM_words)
        LM_words = self.LM.lm_head(LM_words)
        
        ### Gated cross attention: FUSE LM raw output & VPT-transformer-layer-1 output
        # fusing back with VPT-transformer-layer-1 output before going into VPT layer 2 (and using gated cross-attention) means that at the start of training, despit the added LM, the VPT model is identical to the unmodified version, and interaction between the two models can smoothly increase as it is learnt to be useful (as in Flamingo). allows stable training and avoid catastrophic forgetting.
        # so language is passed as queries and VPT tokens are passed at keys/values.
        Xattn_mask=th.ones([num_frames, num_words]) # create causal mask between language tokens and frames. 
        for q in range(num_frames):
            Xattn_mask[q, 0:q+1] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.
        x = self.Xattn_LM_VPT(x=fused, y=x, attn_mask=Xattn_mask) #now that word/frame queries/keys have beens waped, we need to make a different attention mask to the first one
        # ----------------------------------------- END INSERT LM
        print('fused output=',x.shape)



        # pass combined tokens through remaining VPT transformer layers
        x, VPT234_state_out = self.recurrent_layer(x, first, VPT_state_in[1:4], start_block=1, end_block=4)
        VPT_state_out = (VPT1_state_out[0], VPT234_state_out[0], VPT234_state_out[1], VPT234_state_out[2])

        # VPT PREDICT ACTION (for behaviour modelling loss)
        x = self.lastlayer(x)
        x = self.final_ln(x)

        # format action and output
        pi_latent = vf_latent = x

        print('single out=',self.single_output)
        print('x=',x.shape)

        if self.single_output:
            return pi_latent, LM_words, VPT_state_out

        if inference:
            return (pi_latent, vf_latent), LM_words, VPT_state_out, LM_state_out, past_xattn

        return (pi_latent, vf_latent), LM_words, VPT_state_out
            

    def initial_state(self, batchsize):
        return self.recurrent_layer.initial_state(batchsize)

# NOTE OF HOW FORWARD ABOVE WORKS
# pass in a trajectory of consecutive frames and their target actions.
# VPT then convolves attention across the frames causally up to 128 tokens in past, e.g. token0 is responsible for action at 0 and has access to frames 0
# token 10 is responsible for action 10 and can see frames 0 through 10, token 9 for 9 and sees 0 through 9.
# token 128 sees from 0 to 128, token 129 sees from 1 to 129, 256 from 128 to 256
# We can even append a frame/target_action sequence to the same batch using the 'first' tensor (AKA 'context' here) to indicate when the new sequence starts so that the causal mask can be created appropriately to not allow tokens to see frames across different sequences. 
#
# In this implementation, we will have sequences up to the max context length of the LM (2048)
# and ensure that the first langauge embedding is always set to BOS token
# This way, as we randomly sample sequences from across the dataset, we can get the LM to see truncates text sequences at any point in an episode
#
# During inference, we can then let the LM predict one token at a time and use past_key_values to allow for quick inference until it gets to
# the context limit 2048. at this point, at every new token (2049 inputs), the input words need to be cropped by 1 to the last 2048 tokens (2048) and the first token needs to be set to BOS.
# Thsi replicates the training on trunccate sentences, allowing for prediction to continue. However, this is very slow now, as now that inputs are being shifted, we cannot use
# past keys, and all past input word tokens (2047) need to be re-computed so taht the new incoming token (2048th) has proper context.
# this means, during inference, if the language observations grow past 2048, we need to do 2048 LM token passes for every new action.
# Maybe this could be sped up by doing LM_past_keys manipulation?
# Another advantage helping is that since VPT is recurrent, LM can see states even further back, but not langauge further back (unless transformerXL recurrence allows later layers to output data to previous layers somehow).
#
# since most frames have silence tokens. although passing the frame through the LM may allow the LM to give more input which may be useful, its probably best to
# only have it interact every D frames, roughly corresponding to the WPM, so that it interacts only when there is a token. This cannot be guaranteed - if two tokens occur right near each other it will have to output two tokens at that interaction step. ACTUALLY THIS ALREADY HAPPENS FOR EACH WORD. maybe it doesnt matter - simply add missed tokens to stack and output linearly. reduces silence tokens further and probably close enough to actual langauge speed? not a causal problem - bringing past into future, not future into past. LM will output speech at a weirdly regular rate though and may say some things late, hopefully the long term behaviour is preserved though.
# if is LM interacting every D frames, use input frame sequence length of 2048*D, this way the LM has its context size full every sample ,since it only works on every D frames. (outputs are duplicated from last active tiemstep until new LM-active timestep)
#
# NOTE: transformerXL is basically just continually passing past_keys from transformer forward. policy(A) -> VPT_state_out -> policy(B,VPT_state_out)   =  policy(A,B)
# you cane stimate any length of sequence and each new token pass is like pass ing VPT_past_staet through each new token pass.

# AVERAGES RECORDED: tokens/minute: 186 (over 3 hours)     = 322 ms/tok
# AVERAGES RECORDED: tokens/minute: 231 (over 18 mins)
# AVERAGES RECORDED: tokens/minute: 293 (over 21 mins)     = 204 ms/tok
# AVERAGES RECORDED: tokens/minute: 192 (over 27 mins)
# AVERAGES RECORDED: tokens/minute: 229 (over 1 hour)
# average =         # 186+231+293+192+229 = 226 tokens/ minute average   =  265ms/token
# at @20Hz, 50ms/frame
# @d=1: millisecond gap between interaction = 50ms*1 = 50ms     = 50ms*2048 = 102s history
# @d=3: millisecond gap between interaction = 50ms*3 = 150ms    = 307s history
# @d=4: millisecond gap between interaction = 50ms*4 = 200ms    = 409s history = 7 mins history  <--- if we define an objective in langauge at the start, we will need to continuously provide this



























class MinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs, device):
        super().__init__()
        self.net = MinecraftPolicy(device, **policy_kwargs)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)

        self.device=device

        # for keeping internal track of tokens during inference
        self.LM_state_in=None
        self.past_xattn = None
        self.word_context = None
        self.frame_n = 0

    def make_value_head(self, v_out_size: int, norm_type: str = "ewma", norm_kwargs: Optional[Dict] = None):
        return ScaledMSEHead(v_out_size, 1, norm_type=norm_type, norm_kwargs=norm_kwargs)

    def make_action_head(self, pi_out_size: int, **pi_head_opts):
        return make_action_head(self.action_space, pi_out_size, **pi_head_opts)

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()
        self.value_head.reset_parameters()

    def forward(self, ob_words, ob_frames, first: th.Tensor, VPT_state_in, inference=False,
                                                                            LM_state_in=None,
                                                                            past_xattn=None,
                                                                            LM_active=True):
        # extract mask from ob_frames
        if isinstance(ob_frames, dict):
            # We don't want to mutate the obs input.
            ob_frames = ob_frames.copy()
            # If special "mask" key is in obs,
            # It's for masking the logits.
            # We take it out (the network doesn't need it)
            mask = ob_frames.pop("mask", None)
        else:
            mask = None

        if inference:
            (pi_h, v_h), pd_word, VPT_state_out, LM_state_out, past_xattn = self.net(
                                                            ob_words=ob_words, 
                                                            ob_frames=ob_frames, 
                                                            VPT_state_in=VPT_state_in, 
                                                            context={"first": first},
                                                            inference=inference,
                                                            LM_state_in=LM_state_in,
                                                            past_xattn=past_xattn,
                                                            LM_active=LM_active
                                                            )
        else:
            (pi_h, v_h), pd_word, VPT_state_out = self.net( ob_words=ob_words, 
                                                            ob_frames=ob_frames, 
                                                            VPT_state_in=VPT_state_in, 
                                                            context={"first": first})
        pi_logits = self.pi_head(pi_h, mask=mask)
        vpred = self.value_head(v_h)

        if inference:
            return (pi_logits, vpred, None), pd_word, VPT_state_out, LM_state_out, past_xattn
        else:
            return (pi_logits, vpred, None), pd_word, VPT_state_out

    def get_logprob_of_action(self, pd, action):
        """
        Get logprob of taking action `action` given probability distribution
        (see `get_gradient_for_action` to get this distribution)
        """
        ac = tree_map(lambda x: x.unsqueeze(1), action)
        log_prob = self.pi_head.logprob(ac, pd)
        assert not th.isnan(log_prob).any()
        return log_prob[:, 0]

    def get_kl_of_action_dists(self, pd1, pd2): #@ use between old LM (ideally pre-computed) and training LM   # since new input, KL loss wont work well, either training on bad KL target or training to ignore performance. besides, silence tokens are now sorted to be light enough to allow for a  dense langauge signal, so langaueg shouldnt be catastroiphically forgotten now
        """
        Get the KL divergence between two action probability distributions
        """
        return self.pi_head.kl_divergence(pd1, pd2)


    def get_output_for_observation(self, ob_words, ob_img, VPT_state_in, first):
        """
        Return gradient-enabled outputs for given observation.

        Use `get_logprob_of_action` to get log probability of action
        with the given probability distribution.

        Returns:
          - probability distribution given observation
          - value prediction for given observation
          - new state
        """
        ob_img = tree_map(lambda x: x.unsqueeze(1), ob_img)
        first = first.unsqueeze(1)
        
        (pd_action, vpred_action, _), pd_word, VPT_state_out = self(  ob_frames=ob_img,
                                                                                                ob_words=ob_words, 
                                                                                                first=first, 
                                                                                                VPT_state_in=VPT_state_in)

        return pd_action, self.value_head.denormalize(vpred_action), pd_word, VPT_state_out



    def get_output_for_observations(self, ob_words, ob_frames, VPT_state_in=None, dummy_first=None):
        """
        Return gradient-enabled outputs for a sequence of frames.

        `video_frames` should be of shape (N, H, W, C).
        Returns MineRL action dict, where each action head
        has shape (N, ...).

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        if dummy_first == None:
            dummy_first = th.zeros((ob_frames['img'].shape[0], ob_frames['img'].shape[1]), dtype=th.bool).to(self.device)
            #dummy_first[:,0]=True
            #dummy_first = th.zeros((ob_frames['img'].shape[0], 1), dtype=th.bool).to(self.device)

        # set state to zero
        if not VPT_state_in:
            VPT_state_in = self.initial_state(1)

        # pass through agent NN
        (pd_action, vpred_action, _), pd_word, VPT_state_out = self(
            ob_words=ob_words,
            ob_frames=ob_frames,
            first=dummy_first,
            VPT_state_in=VPT_state_in,
        )

        return pd_action, self.value_head.denormalize(vpred_action), pd_word, VPT_state_out




    # RL inference
    # you must pass in starter words as [2, 124, 1256, 77855, 7258]
    ## @@@@ EDIT TO CONFORM TO LM_TIMEOUT A.K.A 'D'
    @th.no_grad()
    def act(self, obs_frame, first, VPT_state_in, stochastic: bool=True, LM_state_in=None, taken_action=None, return_pd=False, starter_words=None):
        # we can feed LM_state_in as none if no prevous state exists.
        LM_TIMEOUT=4

        if not self.word_context:
            if starter_words:
                self.word_context = th.tensor(starter_words).reshape([1,len(starter_words)], dtype=th.bool)
            else:
                self.word_context = th.full([1,1],2, dtype=th.bool)
        if not self.past_xattn:
            self.past_xattn = th.tensor([1,0,1024])   # keep track of past LM inputs in case LM goes past context limit uring inference and they need to be re-computed

        # We need to add a fictitious time dimension everywhere, since during RL inference every action is just one token pass but Agent can take multiple tokens (it does when doing inference since tokens are stored passed to itself through time through saved keys i.e. the TransformerXL mechanism)
        self.frame_context = tree_map(lambda x: x.unsqueeze(1), obs_frame)
        first = first.unsqueeze(1)

        # from words and frames, get next action
        (pd_action, vpred_action, _), pd_word, VPT_state_out, LM_state_out, past_xattn = self(
                                                                    obs_frames=self.frame_context,
                                                                    obs_words=self.word_context[:,-1], # pass most recent word, to go with frame
                                                                    first=first, 
                                                                    VPT_state_in=VPT_state_in,
                                                                    inference=True,
                                                                    LM_state_in=self.LM_state_in,
                                                                    past_xattn = self.past_xattn,
                                                                    LM_active=self.frame_n%LM_TIMEOUT==0
                                                                    )
        
        self.LM_state_in = LM_state_out
        self.past_xattn = past_xattn

        vpred_word = th.argmax(pd_word) # MOST LIKELY WORD: USE ACTUAL SAMPLING # PREDICT FROM: WORD | SILENCE. NOT OVER BOTH @@@@@@@@@@@
        if len(self.word_context) < (self.past_xattn): # do not overwrite starter word context given by user
            self.word_context = th.cat([self.word_context, vpred_word], axis=1)
        # @@ NEED TO SAMPLE FROM ACTUAL BEAM SEARCH FOR GOOD QUALITY OUTPUT, BUT THIS SEARCHES FOR MOST LIKEL SENTENCE AND WE CAN ONLY OUTPUT 1 TOKEN EACH TIMESTEP,
        # AND WE NEED AN OUTPUT EACH TIMESTEP. 
        # MAYBE DO BEAM SEARCH AND THEN OUTPUT JUST FIRST TOKEN FROM THE RESULT?
        # IF NEW FRAMES DONT CHANGE TOO MUCH ASKING THE LM AT THE NEXT TIMESTEP SHOULD BASICALLY GIEV THE SAME PREDITION
        # BUT IF DIFFERENT SHOULD ACT DIFFERENTLY, MAYBE EVEN COMPLETELY CHANIGN TOPIC - INTERRUPTing ITSELF - human-like behaviour?
        self.current_ms += 1

        if taken_action is None:
            ac = self.pi_head.sample(pd_action, deterministic=not stochastic)
        else:
            ac = tree_map(lambda x: x.unsqueeze(1), taken_action)
        log_prob = self.pi_head.logprob(ac, pd_action)
        assert not th.isnan(log_prob).any()

        # After unsqueezing, squeeze back to remove fictitious time dimension
        result = {"log_prob": log_prob[:, 0], "vpred": self.value_head.denormalize(vpred_action)[:, 0]}
        if return_pd:
            result["pd"] = tree_map(lambda x: x[:, 0], pd_action)
        ac = tree_map(lambda x: x[:, 0], ac)

        return ac, VPT_state_out, result    # add newly predicted word to contexxtm which should be fed back into the agent.


    @th.no_grad()
    def v(self, obs, first, state_in):
        """Predict value for a given mdp observation"""
        
        (pd_action, vpred, _), pd_word, state_out = self(obs=obs, first=first, state_in=state_in)

        return self.value_head.denormalize(vpred)




































class InverseActionNet(MinecraftPolicy):
    """
    Args:
        conv3d_params: PRE impala 3D CNN params. They are just passed into th.nn.Conv3D.
    """

    def __init__(
        self,
        hidsize=512,
        conv3d_params=None,
        device=None,
        **MCPoliy_kwargs,
    ):
        super().__init__(
            device=device,
            hidsize=hidsize,
            # If we're using 3dconv, then we normalize entire impala otherwise don't
            # normalize the first impala layer since we normalize the input
            first_conv_norm=conv3d_params is not None,
            **MCPoliy_kwargs,
        )
        self.conv3d_layer = None
        if conv3d_params is not None:
            # 3D conv is the first layer, so don't normalize its input
            conv3d_init_params = deepcopy(self.init_norm_kwargs)
            conv3d_init_params["group_norm_groups"] = None
            conv3d_init_params["batch_norm"] = False
            self.conv3d_layer = FanInInitReLULayer(
                layer_type="conv3d",
                log_scope="3d_conv",
                **conv3d_params,
                **conv3d_init_params,
            )

    def forward(self, ob, state_in, context):
        first = context["first"]
        x = self.img_preprocess(ob["img"])

        # Conv3D Prior to Impala
        if self.conv3d_layer is not None:
            x = self._conv3d_forward(x)

        # Impala Stack
        x = self.img_process(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)

        x = F.relu(x, inplace=False)

        pi_latent = self.lastlayer(x)
        pi_latent = self.final_ln(x)
        return (pi_latent, None), state_out

    def _conv3d_forward(self, x):
        # Convert from (B, T, H, W, C) -> (B, H, W, C, T)
        x = transpose(x, "bthwc", "bcthw")
        new_x = []
        for mini_batch in th.split(x, 1):
            new_x.append(self.conv3d_layer(mini_batch))
        x = th.cat(new_x)
        # Convert back
        x = transpose(x, "bcthw", "bthwc")
        return x




















class InverseActionPolicy(nn.Module):
    def __init__(
        self,
        action_space,
        pi_head_kwargs=None,
        idm_net_kwargs=None,
    ):
        super().__init__()
        self.action_space = action_space

        self.net = InverseActionNet(**idm_net_kwargs)

        pi_out_size = self.net.output_latent_size()

        pi_head_kwargs = {} if pi_head_kwargs is None else pi_head_kwargs

        self.pi_head = self.make_action_head(pi_out_size=pi_out_size, **pi_head_kwargs)

    def make_action_head(self, **kwargs):
        return make_action_head(self.action_space, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        self.pi_head.reset_parameters()

    def forward(self, obs, first: th.Tensor, state_in, **kwargs):
        if isinstance(obs, dict):
            # We don't want to mutate the obs input.
            obs = obs.copy()

            # If special "mask" key is in obs,
            # It's for masking the logits.
            # We take it out (the network doesn't need it)
            mask = obs.pop("mask", None)
        else:
            mask = None

        (pi_h, _), state_out = self.net(obs, state_in=state_in, context={"first": first}, **kwargs)
        pi_logits = self.pi_head(pi_h, mask=mask)
        return (pi_logits, None, None), state_out

    @th.no_grad()
    def predict(self, obs, deterministic: bool = True, **kwargs,):
        (pd, _, _), state_out = self(obs=obs, **kwargs)

        ac = self.pi_head.sample(pd, deterministic=deterministic)
        log_prob = self.pi_head.logprob(ac, pd)

        assert not th.isnan(log_prob).any()

        result = {"log_prob": log_prob, "pd": pd}

        return ac, state_out, result

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)































""" --------------- LATERNATIVE WAY TO FUSE LM



 every frame, we need to have data from the new frame going through the language model and into the 
 next VPT layer. This needs to be done even when there is not a new langauge token with a frame.
 Because of how cross attention works in standard casual LM training, this means that a frame without a corresponding langauge token can not
 be cross attended with.
e.g. when the last word was 10 secs ago, how do we get the latest fame into the LM when it has no new langauge token
to be attended by?

we can either:
    - re-compute old words with new frames: warning: this involves re-computing all langauge tokens
    for every new frame coming in, so that ['hi', F0,F1,F2] and ['hi', F4,F5,F6] both have all frames being attended to
    - With WPM of 150 and context of 2048, we can see 2048 language tokens reaching back about 10.2 minutes

    - add silence tokens so that every new frame has a language token to attend to it. this is far more efficient and faster,
    every frame only requires only 1 pass through a the LM to be done as opposed to having to re-calculate all old langauge tokens too.
    HOWEVER: this also means that silence tokens now take up the majority of the 2048 token context, 
    both creating a lot of noise inthe dataset that is very different from LM pretraining and may result in cmopromised performance,
    and also, at e.g. 150WPM, only 16% of tokens are actual language signals, the rest is silence, so at 150WPM we can
    only see about 350 tokens each pass, and only see about 2 minutes into the past
    - may be do this in a way such that every other frame is copied from the previous. This way 1/2 the silence tokens are used, besides alngauge is probably slow
    - doing this langauge-token-per-frame, and reduce rate of langauge model interaction (repeating rpevious output until next) is much easier to implement and clearner, proably more performant.
        only issue is how to deal with: 2 words said during blank interval. now langauge model takes in previous 1/2 frames and which langauge token?















"""