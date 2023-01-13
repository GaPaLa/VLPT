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
    
    def forward(self, ob_words, ob_frames, VPT_state_in, context):

        #@ for both ms, ms should be a [num_frames] or [num_words] long array
        #@ change word_obs  to allow for batching @batch1

        first = context["first"]
        print("agent.policy.net.forward(): ob_words, ob_frames shapes:",  ob_words.shape, ob_frames['img'].shape )

        

        x = self.img_preprocess(ob_frames["img"])
        x = self.img_process(x)

        # pass frame through VPT Agent transformer layer #1
        x, VPT1_state_out = self.recurrent_layer(x, first, [VPT_state_in[0]], start_block=0, end_block=1)
        

        # -------------------------------- INSERT LM

        x2_ = x.clone()
        #only call once per batch.
        
        # for every frame, pass previous 128 frames through LM
        LM_out = []*x2.shape[1]
        word_out = []
        for t in range(len(x2_.shape[1])):
            

            # get past frames from t up to 128 before
            frame_ms = ob_frames['ms'][t]

            # get words from frame t backwards
            token_ids_in_context = ob_words['token_ids'][th.where(ob_words['ms']<=frame_ms)]
            # crop at 2048 words
            token_ids_in_context = token_ids_in_context[max(-2048, -len(token_ids_in_context)):]
            #@ make sure to add word padding so sentences with different lengths work
            token_ids_padding = 

            # frames in context. get last 128 from selected frame
            # if referencing index -n, crop to first frame
            x2 = x2_[max(-2048, -x2.shape[1]):t]
            


                


            # convert word token ids to embeddings. since LM(words) works from token_indices and we need to pass
            # VPT representations as raw vectors, we need to use LM(words, from_embeddings=True),
            # which requires us to pre-embed to token ids
            l = self.LM.model.decoder.embed_tokens(token_ids_in_context)      # for more performance, edit so that we work on ful LM embeddings (1024) rather then word representations (512), i.e. work just before positional embeddings are added, which are also 1024 big.
            l = self.LM.model.decoder.project_in(l)

            # FUSE VPT_transformer_layer_1_output & language_input_tokens via gated-cross-attention. PASS into LM
            # PASS VPT_transformer_layer_1 OUTPUT AND WORDS THROUGH LM
            Xattn_mask = th.zeros(l.shape[1], x.shape[1]).to(device) #need to mask future langauge tokens to be predicted in Xattn since this is the first attention layer they pass through
            Xattn_mask[-1,:] = 1
            l = self.Xattn_VPT_LM(x=x2, y=l, attn_mask=Xattn_mask) # repeat for each timestep

            # LM PREDICT OUTPUTS (for word prediction and to give to agent)
            l = self.LM.model.forward(inputs_embeds=l, project_embeds=False, token_ids_padding)['last_hidden_state'] # OPT has causal masking built-in, but need to add padding mask for batch_size>1 @batch1.   # OPT FILES HAVE BEEN MODIFIED. USING 350M AND OTHERS, MODEL EMBEDDING SIZE IS 1024, BUT WORDS EMBEDDINGS INPUTS ARE TO BE 512 AND CAST UP AT INPUT. FOR BETTER PERFORMANCE< NEED TO HAVE ACCESS TO FULL MODEL EMBEDDING SIZE SO VPT INFO (2048) DOESNT GET SQUISHED TO 512 THEN SCALED UP AGAIN. ADDED INPUT TO FORWARD CALLED 'project_embeds' to make model accept  full embeddings. requires projecting words embedding earlier, but  after that we have access to modifying that to pass in full 1024 embeddings
            LM_out[t] = l[:,-1,:]    # CROSS ATTENTION ALL LM TOKENS OUT WITH VPT OR JUST LAST - JUST LAST, past tokens have already had their chance to be last, if info important it can be rereouted #@batch1 - adjust for padding

            # LM PREDICT WORD (for language modelling loss)
            LM_head = l.clone()
            LM_head = self.LM.model.decoder.project_out(LM_head)
            LM_head = self.LM.lm_head(LM_head)
            word_out[t] = LM_head[:,-1,:]    # word out is last word of all batches, get full embedding. @ FIX - last word may just be padding. ADJUST FOR PADDING @batch1

        LM_out_t = th.asarray(LM_out).to(device)
        word_pd = th.asarray(word_out).to(device)

        # GATED CROSS ATTENTION: LM OUTPUT AND VPT-transformer-layer-1 OUTPUT
        # Since this input is going to VPT, ans it is important that the input it sees as init is no diffrent from vanilla VPT.
        # so language is passed as queries and VPT tokens are passed at keys/values.
        x = self.Xattn_LM_VPT(x=LM_out_t, y=x)

        # -------------------------------- INSERT LM





        # pass combined tokens through remaining VPT transformer layers
        x, VPT34_state_out = self.recurrent_layer(x, first, VPT_state_in[1:4], start_block=1, end_block=4)
        VPT_state_out = [VPT1_state_out, VPT34_state_out[0], VPT34_state_out[1], VPT34_state_out[2]]

        # VPT PREDICT ACTION (for behaviour modelling loss)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, word_pd, VPT_state_out
        return (pi_latent, vf_latent), word_pd, VPT_state_out


    def initial_state(self, batchsize):
        return self.recurrent_layer.initial_state(batchsize)



















class MinecraftAgentPolicy(nn.Module):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs, device):
        super().__init__()
        self.net = MinecraftPolicy(**policy_kwargs)

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

    def forward(self, ob_words, ob_frames, first: th.Tensor, VPT_state_in):

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

        (pi_h, v_h), pd_word, VPT_state_out = self.net(
                                                        ob_words=ob_words, 
                                                        ob_frames=ob_frames, 
                                                        VPT_state_in=VPT_state_in, 
                                                        context={"first": first})

        pi_logits = self.pi_head(pi_h, mask=mask)
        vpred = self.value_head(v_h)

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

    def get_kl_of_action_dists(self, pd1, pd2):
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



    def get_output_for_observations(self, ob_words, ob_frames):
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
        dummy_first = th.zeros((ob_frames['img'].shape[0], ob_frames['img'].shape[1]), dtype=th.bool).to(self.device)

        # set state to zero
        state_in = self.initial_state(1)

        # pass through agent NN
        (pd_action, vpred_action, _), pd_word, VPT_state_out = self(
            ob_words=ob_words,
            ob_frames=ob_frames,
            first=dummy_first,
            VPT_state_in=state_in,
        )

        return pd_action, self.value_head.denormalize(vpred_action), pd_word






    # RL inference
    @th.no_grad()
    def act(self, obs_frame, obs_word, first, VPT_state_in, stochastic: bool=True, taken_action=None, return_pd=False):
        # We need to add a fictitious time dimension everywhere, since during RL inference every action is just one token pass but Agent expects multiple tokens
        obs = tree_map(lambda x: x.unsqueeze(1), obs)
        first = first.unsqueeze(1)

        (pd_action, vpred_action, _), pd_word, VPT_state_out = self(
                                                                    obs_frame=obs_frame, 
                                                                    obs_word=obs_word, 
                                                                    first=first, 
                                                                    VPT_state_in=VPT_state_in
                                                                    )
        
        vpred_word = th.argmax(pd_word) # MOST LIKELY WORD: USE ACTUAL SAMPLING # PREDICT FROM: WORD | SILENCE. NOT OVER BOTH @@@@@@@@@@@

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

        return ac, vpred_word, VPT_state_out, result


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
        **MCPoliy_kwargs,
    ):
        super().__init__(
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
    def predict(
        self,
        obs,
        deterministic: bool = True,
        **kwargs,
    ):
        (pd, _, _), state_out = self(obs=obs, **kwargs)

        ac = self.pi_head.sample(pd, deterministic=deterministic)
        log_prob = self.pi_head.logprob(ac, pd)

        assert not th.isnan(log_prob).any()

        result = {"log_prob": log_prob, "pd": pd}

        return ac, state_out, result

    def initial_state(self, batch_size: int):
        return self.net.initial_state(batch_size)































""" --------------- LATERNATIVE WAY TO FUSE LM



 # -------------------------------- INSERT LM INTO VPT
 every frame, we need to have data from the new frame going through the language model and into the 
 next VPT layer. This needs to be done even when there is not a new langauge token with a frame,
 Because of how cross attention works, this means that the frame can nor cross attend with any new token in
, compared to how when a new token comes into a regular self attention layer it always gets attention.
e.g. when the last word was 10 secs ago, how do we get the latest fame into the LM when it has no new langauge token
to be attended by?

we can either:
    - re-compute old words with new frames: warning: this involves re-computing all langauge tokens
    for every new frame coming in, so that ['hi', F0,F1,F2] and ['hi', F4,F5,F6] both have all frames being attended to
    - With WPM of 150 and context of 2048, we can see 2048 language tokens reaching back about 10.2 minutes

    - add silence tokens so that every new frame has a language token to attend to it. this is far more efficient and faster,
    requiring only 1 pass through a transformer as opposed to re-calculating all old tokens.
    HOWEVER: this also means that silence tokens now take up the majority of the 2048 token context, 
    both creating a lot of noise inthe dataset that is very different from LM pretraining and may result in cmopromised performance,
    and also, at e.g. 150WPM, only 16% of tokens are actual language signals, the rest is silence, so at 150WPM we can
    only see about 350 tokens each pass, and only see about 2 minutes into the past

        # frames and words MUST be passed matched. e,g. if frames = [f0,f1,f2,f3]
                                                            words = [w0,w1,w2,w3]

        x2_ = x.clone()
        #only call once per batch.
        
        # it would be nice to be able to pad out silence tokens form the inpout to not ditort the input but
        # the whole point is that every frame information is going through from teh frames, so silence tokens
        # should not be masked

        # get words from frame t backwards
        token_ids_in_context = ob_words['token_ids'][th.where(ob_words['ms']<=frame_ms)]
        # crop at 2048 words
        token_ids_in_context = token_ids_in_context[max(-2048, -len(token_ids_in_context)):]
        # make sure to add word padding so sentences with different lengths work

        # frames in context. get last 128 from selected frame
        # if referencing index -n, crop to first frame
        x2 = x2_[max(-2048, -x2.shape[1]):t]
        


            


        # convert word token ids to embeddings. since LM(words) works from token_indices and we need to pass
        # VPT representations as raw vectors, we need to use LM(words, from_embeddings=True),
        # which requires us to pre-embed to token ids
        l = self.LM.model.decoder.embed_tokens(token_ids_in_context)      # for more performance, edit so that we work on ful LM embeddings (1024) rather then word representations (512), i.e. work just before positional embeddings are added, which are also 1024 big.
        l = self.LM.model.decoder.project_in(l)

        # FUSE VPT_transformer_layer_1_output & language_input_tokens via gated-cross-attention. PASS into LM
        # PASS VPT_transformer_layer_1 OUTPUT AND WORDS THROUGH LM
        #need CAUSAL ATTENTION MASK in Xattn since this is the first attention layer they pass through
        Xattn_mask = th.zeros(l.shape[1], x.shape[1]) 
        Xattn_mask[-1,:] = 1
        l = self.Xattn_VPT_LM(x=x2, y=l, attn_mask=Xattn_mask) # repeat for each timestep

        # LM PREDICT OUTPUTS (for word prediction and to give to agent)
        l = self.LM.model.forward(inputs_embeds=l, project_embeds=False, attention_mask=ob_words['attention_mask'] )['last_hidden_state'] # since future langauge tokens are masked in x-attention, we don't have to mask them here. IF WE DO, LM USES OPPOSITE ASKING SCHEME TO X-ATTENTION. USE LM_mask = (language_maks=0)
        LM_out[t] = l[:,-1,:]    # CROSS ATTENTION ALL LM TOKENS OUT WITH VPT OR JUST LAST - JUST LAST, past tokens have already had their chance to be last, if info important it can be rereouted #@batch1 - adjust for padding

        # LM PREDICT WORD (for language modelling loss)
        LM_head = l.clone()
        LM_head = self.LM.model.decoder.project_out(LM_head)
        LM_head = self.LM.lm_head(LM_head)
        word_out[t] = LM_head[:,-1,:]    # word out is last word of all batches, get full embedding. @ FIX - last word may just be padding. ADJUST FOR PADDING @batch1

        # GATED CROSS ATTENTION: LM OUTPUT AND VPT-transformer-layer-1 OUTPUT
        # Since this input is going to VPT, ans it is important that the input it sees as init is no diffrent from vanilla VPT.
        # so language is passed as queries and VPT tokens are passed at keys/values.
        x = self.Xattn_LM_VPT(x=LM_out_t, y=x)

        # -------------------------------- INSERT LM













"""