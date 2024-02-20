from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math 
import pdb
from typing import Dict, Any

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
                         

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from transformers.cache_utils import Cache, DynamicCache


import sys
from allava.model.language_model.phi.modeling_phi import PhiForCausalLM, PhiModel, PhiConfig




################ Phi ###############################

class LlavaPhiConfig(PhiConfig):
    model_type = "llava_phi"

class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = LlavaPhiConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)



class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config, init_vision_encoder_from_ckpt=False):
        config._attn_implementation = "flash_attention_2"

        super(PhiForCausalLM, self).__init__(config)
        # self.model is used in LlavaMetaForCausalLM.get_model(); self.transformer is used in PhiForCausalLM.forward()
        self.model = LlavaPhiModel(config)
        if hasattr(self.model, '_use_flash_attention_2'):
            assert self.model._use_flash_attention_2, 'flash attn is not enabled. check it out!'
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if init_vision_encoder_from_ckpt:
            vision_tower = self.get_vision_tower()
            print(f'loading from CLIP first. This should only be used at inference!!!')
            vision_tower.load_model() # 
            
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            # ) = self.prepare_inputs_labels_for_multimodal(
            ) = self.prepare_inputs_labels_for_multimodal_new(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # pdb.set_trace()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        '''
        This function is called for each token at inference
        '''
        # pdb.set_trace()
        images = kwargs.pop("images", None)

        ####################################################
        # lines from modeling_phi.py
        ####################################################

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif past_length >= input_ids.shape[1]:
                input_ids = input_ids[:, [-1]] # only keep the last one!

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        ####################################################
        # end of lines from modeling_phi.py
        ####################################################


        if images is not None:
            model_inputs['images'] = images
        return model_inputs


    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
    #     '''
    #     This function is called for each token at inference
    #     '''
    #     pdb.set_trace()
    #     images = kwargs.pop("images", None)

        
    #     _inputs = super().prepare_inputs_for_generation(
    #         input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
    #     )
    #     if images is not None:
    #         _inputs['images'] = images
    #     return _inputs


AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)