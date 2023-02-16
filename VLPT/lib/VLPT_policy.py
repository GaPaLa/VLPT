from transformers import TransfoXLLMHeadModel

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
        SEQ_LEN,
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

        self.SEQ_LEN = 128
        self.LM_TIMEOUT=2# referred to also as 'D'. LM gets input from VPT and gives output to VPT every D timesteps to reduce silence token noise.
        LM_tgt_len=256 # determined by input size


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
            **impala_kwargs)
        




        # Define Language Model
        self.LM = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103") #, @later, load from stored weights
        self.LM.tie_weights() # necessary for this model)
        # @ CHANGE THESE DURING INFERENCE
        LM_mem_len=LM_tgt_len # number of previous keys to cache. eval:4608 = 256*18
        self.LM.transformer.config.tgt_len = LM_tgt_len
        self.LM.transformer.config.mem_len = LM_mem_len 
        self.LM.transformer.config.same_len = False # eval:True
        self.LM.transformer.config.clamp_len = -1 # eval: 2000 - definitely experiment.





        # define cross atention layers from VPT->LM, LM->VPT
        self.Xattn_VPT_LM = MaskedGatedCrossAttention(
            embed_dim=self.LM.transformer.d_embed, # queries are lanuage tokens - LM gets appropriate size and number of tokens 
            kvdim=hidsize, # embedding dimensions for VPT transformer layers
            num_heads=attention_heads,
            ffw_dim=hidsize*pointwise_ratio, # FFW hidden size should be same architecturally as FFW in LM (i.e. ratio between transformer token size and FFW hidden layer should be same) (x2 to account for multimodal tokens)
            batch_first=True,
            dtype=th.float32)
        self.Xattn_LM_VPT = MaskedGatedCrossAttention(
            embed_dim=hidsize, # queries are VPT tokens - VPT gets appropriate size and number of tokens
            kvdim=self.LM.transformer.d_embed,
            num_heads=attention_heads,
            ffw_dim=hidsize*pointwise_ratio, # use same as FFW (x2 to account for multimodal tokens)
            batch_first=True,
            dtype=th.float32)

        # for speed, construct attention masks here
        # cross atention between every D langaueg tokens and every frame
        self.Xattn_mask_in=th.ones([self.SEQ_LEN//self.LM_TIMEOUT, self.SEQ_LEN]) # create causal mask between language tokens and frames. every frame is paired with the previous word emitted, so that at the current frame we predic the next word from the current frame (and previous frames witha ttentions) and rpevious words. For more details look at agent.py words_to_agent() # since all sequeces in the batch (and every batch assuming constant D) have the same relation in terms of time with each other, all sequences can use the same Xattn mask
        for q in range(self.SEQ_LEN//self.LM_TIMEOUT):
            self.Xattn_mask_in[q, 0:q+self.LM_TIMEOUT] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.
        # cross attention between every langauge output (including repeated tokens from LM durin LM timeout) and every frame
        self.Xattn_mask_out=th.ones([self.SEQ_LEN, self.SEQ_LEN]) # create causal mask between language tokens and frames. every frame is paired with the previous word emitted, so that at the current frame we predic the next word from the current frame (and previous frames witha ttentions) and rpevious words. For more details look at agent.py words_to_agent() # since all sequeces in the batch (and every batch assuming constant D) have the same relation in terms of time with each other, all sequences can use the same Xattn mask
        for q in range(self.SEQ_LEN):
            self.Xattn_mask_out[q, 0:q+1] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.








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
    def forward(self, ob_words, ob_frames, VPT_state, context, LM_state=None, LM_active_timestep=True, LM_labels=None):
        





        ############################ ugly preprocesing thats in a bad place please ignore this :)
        if ob_words.shape[1] != self.SEQ_LEN:
            t_seq_len = ob_words.shape[1]
            # for speed, construct attention masks here
            # cross atention between every D langaueg tokens and every frame
            self.Xattn_mask_in=th.ones([max(1,t_seq_len//self.LM_TIMEOUT), t_seq_len]) # create causal mask between language tokens and frames. every frame is paired with the previous word emitted, so that at the current frame we predic the next word from the current frame (and previous frames witha ttentions) and rpevious words. For more details look at agent.py words_to_agent() # since all sequeces in the batch (and every batch assuming constant D) have the same relation in terms of time with each other, all sequences can use the same Xattn mask
            for q in range(max(1, t_seq_len//self.LM_TIMEOUT)):
                self.Xattn_mask_in[q, 0:q+self.LM_TIMEOUT] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.
            # cross attention between every langauge output (including repeated tokens from LM durin LM timeout) and every frame
            self.Xattn_mask_out=th.ones([t_seq_len, t_seq_len]) # create causal mask between language tokens and frames. every frame is paired with the previous word emitted, so that at the current frame we predic the next word from the current frame (and previous frames witha ttentions) and rpevious words. For more details look at agent.py words_to_agent() # since all sequeces in the batch (and every batch assuming constant D) have the same relation in terms of time with each other, all sequences can use the same Xattn mask
            for q in range(t_seq_len):
                self.Xattn_mask_out[q, 0:q+1] = 0   # @ DOUBLE CHECK THIS IS MASKING THE RIGHT WAY #@r2 - !if using each_LM_token_attends_to_all_past_frames-style x-attn (as opposed to only frames that occured during langauge token), modify this mask appropriately.


























        ## useful stuff for later
        first = context["first"]
        ### pass input frames through CNN section
        x = self.img_preprocess(ob_frames["img"])
        x = self.img_process(x)
        ### pass processed frames through VPT Agent transformer layer #1. make sure only 128 tokens are rpedicted with self attention at a time, otherwise trained differently to original (?) and BIG MEM.
        x, [VPT_state[0]] = self.recurrent_layer(x, first, [VPT_state[0]], start_block=0, end_block=1)


        # ----------------------------------------- INSERT LM
        
        ### copy VPT_transofrmer_layer1 output to pass the LM via cross-attention, leaving original output as residual around LM (as in Flamingo-style gated cross-attention. see: masked_gated_cross_atention.py)
        # implement LM timeout - D
        ### convert input word token ids to embeddings. since LM(words) works from token_indices and we need to pass
        # VPT representations as raw vectors, we need to use LM(words, from_embeddings=True),
        # which requires us to pre-embed to token ids
        x2 = x.clone() # changed to get every frame since due to more efficient mechanism and realise more data=more better #get every Dth frame. means that LM only sees as many framse as words i.e. frames are as spares as words. We CAN train on all past frames every langauge token, however, this results in 2048*D frames cross-attending with language token #2048 as opposed to just 2048 frames corss attending with langauge token #2048. Maybe that's okay? or preferable? It sounds expensive, ill do the sparse frames method. plus, this simplifies the X_attn mask #@r2
        l = ob_words[:,::self.LM_TIMEOUT] # crop word equally. This should have been modified by agent.py . words_to_agent() to only be silent tokens
        l = self.LM.transformer.word_emb(l)
        
        ### Gated cross attention: FUSE output from VPT transformer layer 1 and language input tokens
        # need CAUSAL ATTENTION MASK in cross attention, so langauge tokens cant cross attend to future frames, since the inputted frames and words tensors both have future and past information relative to each other
        # frames and words are ordered such that at frame index 0, word index 0 has already occured. therefore we can take both as input to estimate the next word
        # word 0 can see frame 0 (because frame 0 occurs before the word does).
        x2 = self.Xattn_VPT_LM(x=x2, y=l, attn_mask=self.Xattn_mask_in)

        #### Feed LM fused input tokens & predict raw LM output
        #### From raw LM output, predict word classes (for language modelling loss)
        if LM_active_timestep:

            ### construct labels if possible
            if LM_labels:
                LM_labels = LM_labels[:,::self.LM_TIMEOUT] # since input words are dropped according to D, labels are dropped according to D, too

            ### get LM raw output, word predictions and loss
            LM_words, LM_loss, LM_raw_out, LM_state_out = self.LM(inputs_embeds=x2, mems=LM_state, labels=LM_labels, return_dict='VLPT', output_hidden_states=True) # causal attention mask is constructed internally. No need for padding despite batch_size>1 because all sequences are always full (always same number as input frames)
            # just get last hidden layer output
            x2 = LM_raw_out
            # reshape LM_words
            LM_words = LM_words.view(LM_words.shape[0], x2.shape[1], -1)

        else: # During inferemce when forward() is called per timestep, if LM is TIMEOUT'd at this tiemstep, output previous output from LM that VPT needs (does not include past words)
            LM_words = None # output during TIMEOUT is treated the same as with silent tokens during TIMEOUT, and discarded from LM input. dont need to predict it at all.
            #top layer|all abtches|last token|full embeddings
            x2 = LM_state[-1][:,-1,:] # during inference we can only use the LM every few timesteps. need to tell LM whether to compute at this timestep or just re-use the previous LM output re-use the lastest token from the last layer
            LM_loss = None

        #so that there is an LM output for every frame despite LM_timeout (A.K.A 'D'), repeat every LM outputs D times per step in the time dimension. 
        #Make sure not to output too many values if the input frames is not divisible by D
        x2 = th.repeat_interleave(x2, self.LM_TIMEOUT, dim=1)[:,:self.SEQ_LEN,:] 

        ### Gated cross attention: FUSE LM raw output & VPT-transformer-layer-1 output
        # fusing back with VPT-transformer-layer-1 output before going into VPT layer 2 (and using gated cross-attention) means that at the start of training, despit the added LM, the VPT model is identical to the unmodified version, and interaction between the two models can smoothly increase as it is learnt to be useful (as in Flamingo). allows stable training and avoid catastrophic forgetting.
        # so language is passed as queries and VPT tokens are passed at keys/values.
        x = self.Xattn_LM_VPT(x=x2, y=x, attn_mask=self.Xattn_mask_out) #now that word/frame queries/keys have beens waped, we need to make a different attention mask to the first one
        # ----------------------------------------- END INSERT LM


        # pass combined tokens through remaining VPT transformer layers (2,3,4)
        x, VPT_state[1:4] = self.recurrent_layer(x, first, VPT_state[1:4], start_block=1, end_block=4)

        # VPT PREDICT ACTION (for behaviour modelling loss)
        x = self.lastlayer(x)
        x = self.final_ln(x)

        # format action and output
        pi_latent = vf_latent = x

        return (pi_latent, vf_latent), LM_words, VPT_state, LM_state_out, LM_loss
            

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
    def __init__(self, device, action_space, policy_kwargs, pi_head_kwargs, SEQ_LEN=512):
        super().__init__()
        self.net = MinecraftPolicy(device=device, SEQ_LEN=SEQ_LEN, **policy_kwargs)

        self.action_space = action_space

        self.value_head = self.make_value_head(self.net.output_latent_size())
        self.pi_head = self.make_action_head(self.net.output_latent_size(), **pi_head_kwargs)

        self.device=device

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

    def forward(self, ob_words, ob_frames, first: th.Tensor, VPT_state, LM_state=None, LM_active_timestep=True):
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
        (pi_h, v_h), pd_word, VPT_state_out, LM_state, LM_loss = self.net(
                                                            ob_words=ob_words, 
                                                            ob_frames=ob_frames, 
                                                            VPT_state=VPT_state, 
                                                            context={"first": first},
                                                            LM_state=LM_state,
                                                            LM_active_timestep=LM_active_timestep)
        pi_logits = self.pi_head(pi_h, mask=mask)
        vpred = self.value_head(v_h)

        return (pi_logits, vpred), pd_word, VPT_state_out, LM_state, LM_loss

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


    def get_output_for_observation(self, ob_words, ob_img, VPT_state, first, LM_state=None, LM_active_timestep=True):
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
        
        (pd_action, vpred_action), LM_words, VPT_state_out, LM_state_out, LM_loss = self(  ob_frames=ob_img,
                                                                                                ob_words=ob_words, 
                                                                                                first=first, 
                                                                                                VPT_state=VPT_state,
                                                                                                LM_state = LM_state,
                                                                                                LM_active_timestep=LM_active_timestep)

        return pd_action, self.value_head.denormalize(vpred_action), LM_words, VPT_state_out, LM_state_out, LM_loss



    def get_output_for_observations(self, ob_words, ob_frames, VPT_state=None, LM_state=None, dummy_first=None, LM_active_timestep=True):
        """
        Return gradient-enabled outputs for a sequence of frames.

        `video_frames` should be of shape (N, H, W, C).
        Returns MineRL action dict, where each action head
        has shape (N, ...).

        Agent's hidden state is tracked internally (actually externally as of this class. it is tracked within agent.py). To reset it,
        call `reset()` (in agent.py).
        """
        
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        if dummy_first == None:
            dummy_first = th.zeros((ob_frames['img'].shape[0], min(128,ob_frames['img'].shape[1])), dtype=th.bool).to(self.device)

        # set state to zero
        if not VPT_state:
            VPT_state = self.initial_state(ob_frames['img'].shape[0])

        # pass through agent NN
        (pd_action, vpred_action), LM_words, VPT_state_out, LM_state_out, LM_loss = self(
            ob_words=ob_words,
            ob_frames=ob_frames,
            first=dummy_first,
            VPT_state=VPT_state,
            LM_state=LM_state,
            LM_active_timestep=LM_active_timestep)


        return pd_action, self.value_head.denormalize(vpred_action), LM_words, VPT_state_out, LM_state_out, LM_loss







    # RL inference
    # you must pass in starter words as [2, 124, 1256, 77855, 7258]
    ## @@@@ EDIT TO CONFORM TO LM_TIMEOUT A.K.A 'D'
    @th.no_grad()
    def act(self, obs_frame, first, VPT_state, LM_state, word_context, current_timestep, stochastic: bool=True, taken_action=None, return_pd=False):

        #modify agent for single timestep inference
        tgt_len=self.net.LM.tgt_len
        seq_len=self.net.SEQ_LEN
        self.net.LM.tgt_len=1
        self.net.SEQ_LEN=1
        
        # we can feed LM_state as none if no prevous state exists.
        LM_TIMEOUT=2
        BOS=2

        first = first.unsqueeze(1)
        # We need to add a fictitious time dimension everywhere, since during RL inference every action is just one token pass but Agent can take multiple tokens (it does so when doing inference too despit one at a time, since tokens are stored and passed to itself through time through saved keys (VPT/LM states) i.e. the TransformerXL mechanism)
        current_frame = tree_map(lambda x: x.unsqueeze(1), obs_frame)
        current_word = tree_map(lambda x: x.unsqueeze(1), word_context[-1])
        
        # from words and frames, get next action
        (pd_action, vpred_action, _), pd_word, VPT_state_out, LM_state, LM_loss = self(
                                                                    first=first, 
                                                                    obs_frames=current_frame,
                                                                    obs_words=current_word,  # pass most recent word to pair with current frame. this works even with D: LM outputs something at timestep 0 and attends to it at timestep D and later
                                                                    VPT_state=VPT_state,
                                                                    LM_state=LM_state,
                                                                    LM_active=current_timestep%LM_TIMEOUT==0
                                                                    )

        ### sample next word at this timestep from LM and add back into LM context
        # MOST LIKELY WORD: USE ACTUAL SAMPLING # PREDICT FROM: WORD | SILENCE. NOT OVER BOTH @@@@@@@@@@@
        # @@ NEED TO SAMPLE FROM ACTUAL BEAM SEARCH FOR GOOD QUALITY OUTPUT, BUT THIS SEARCHES FOR MOST LIKEL SENTENCE AND WE CAN ONLY OUTPUT 1 TOKEN EACH TIMESTEP,
        # AND WE NEED AN OUTPUT EACH TIMESTEP. 
        # MAYBE DO BEAM SEARCH AND THEN OUTPUT JUST FIRST TOKEN FROM THE RESULT?
        # IF NEW FRAMES DONT CHANGE TOO MUCH ASKING THE LM AT THE NEXT TIMESTEP SHOULD BASICALLY GIEV THE SAME PREDITION
        # BUT IF DIFFERENT SHOULD ACT DIFFERENTLY, MAYBE EVEN COMPLETELY CHANIGN TOPIC - INTERRUPTing ITSELF - human-like behaviour?
        if pd_word: # word is None if LM not active at this timestep, in which case dont save a word to the word context. This means that the LM can output something at t=0, timeout for D, and then when it runs next, it correctly takes in this previous word as input
            vpred_word = th.argmax(pd_word[0,-1]) # during inference, batch size always == 1, out
            if current_timestep > word_context.shape[1]: # do not overwrite starter word context given by user (or add to the end of it since it that introducess latency between its LM output and input)
                word_context = th.cat([word_context, vpred_word], axis=1)

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



















        tgt_len=self.net.LM.tgt_len
        seq_len=self.net.SEQ_LEN
        self.net.LM.tgt_len=1
        self.net.SEQ_LEN=1



        current_timestep += 1
        return ac, VPT_state_out, result, LM_state, word_context, current_timestep    # add newly predicted word to contexxtm which should be fed back into the agent.






    # not used in VLPT - no Rl
    @th.no_grad()
    def v(self, obs, first, state_in):
        """Predict value for a given mdp observation"""
        
        (pd_action, vpred, _), pd_word, VPT_state_out, LM_state_out, LM_loss = self(obs=obs, first=first, state_in=state_in)

        return self.value_head.denormalize(vpred)

















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











# code used to effectively gradient accumulate VPT predictions to reduce RAM usage. This way VPT can estimate enough states for LM to predict 256 steps in tgt_mem.
       ### pass processed frames through VPT Agent transformer layer #1. make sure only 128 tokens are rpedicted with self attention at a time, otherwise trained differently to original (?) and BIG MEM.
        print('VLPT1:')
        if seq_len>128: # if input sequence is greater than max context length of VPT, do it iteratively in chunks of 128
            assert seq_len%128==0
            x_=[]
            print('seq lrn',seq_len)
            for fwd in range(seq_len//128):
                print('VPT1in:',VPT_state[0])
                print('x in',x[:,fwd*128:(fwd+1)*128,:].shape)
                temp, [VPT_state[0]] = self.recurrent_layer(x[:,fwd*128:(fwd+1)*128,:], first, [VPT_state[0]], start_block=0, end_block=1)
                print('VPT1out:',VPT_state[0])
                print('temp:',temp.shape)
                x_.append(temp.clone())
                del temp
            print('X_:',x_)
            x=th.cat(x_, dim=1)
            print(x)
            del x_
        else:
            x, [VPT_state[0]] = self.recurrent_layer(x, first, [VPT_state[0]], start_block=0, end_block=1)








"""