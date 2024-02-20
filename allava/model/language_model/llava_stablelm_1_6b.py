#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn


from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, PretrainedConfig
                        #  StableLMEpochConfig, StableLMEpochModel, StableLMEpochForCausalLM
from transformers.modeling_utils import cached_file, CONFIG_NAME, extract_commit_hash, is_peft_available, find_adapter_config_file, json, os
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _get_model_class
from transformers.dynamic_module_utils import resolve_trust_remote_code, get_class_from_dynamic_module


from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import pdb

import sys
sys.path.insert(0, '/mntcephfs/data/med/guimingchen/models/general/stablelm-2-1_6b')
from modeling_stablelm_epoch import StableLMEpochForCausalLM, StableLMEpochModel, StableLMEpochConfig


################ stableLM ###############################

class LlavaStableLM_1_6bConfig(StableLMEpochConfig):
    model_type = "llava_stablelm_1_6b"

# class LlavaStableLMModel(LlavaMetaModel, AutoModel):
class LlavaStableLMModel(LlavaMetaModel, StableLMEpochModel):
    config_class = LlavaStableLM_1_6bConfig

    def __init__(self, config: AutoConfig):
        super(LlavaStableLMModel, self).__init__(config)



class LlavaStableLM_1_6bForCausalLM(StableLMEpochForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaStableLM_1_6bConfig


    def __init__(self, config, init_vision_encoder_from_ckpt=False):
        config._attn_implementation = "flash_attention_2"

        super(StableLMEpochForCausalLM, self).__init__(config)

        self.model = LlavaStableLMModel(config)
        if hasattr(self.model, '_use_flash_attention_2'):
            assert self.model._use_flash_attention_2, 'flash attn is not enabled. check it out!'
        # self.pretraining_tp = config.pretraining_tp
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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

# class StableLMEpochConfig = AutoConfig.from_pretrained('/wangbenyou/guimingchen/models/stablelm-3b-4e1t', trust_remote_code=True)


AutoConfig.register("llava_stablelm_1_6b", LlavaStableLM_1_6bConfig)
# AutoConfig.register("stablelm_epoch", LlavaStableLMConfig)
AutoModelForCausalLM.register(LlavaStableLM_1_6bConfig, LlavaStableLM_1_6bForCausalLM)
