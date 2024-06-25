from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pdb

dir = "FreedomIntelligence/ALLaVA-3B-Longer"

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(dir)
model.tokenizer = tokenizer

gen_kwargs = {
    'min_new_tokens': 20,
    'max_new_tokens': 100,
    'do_sample': False,
    'eos_token_id': tokenizer.eos_token_id # this is a must since transformers ~4.37
}

#################################################################################
# first round
#################################################################################
response, history = model.chat(
    texts='What is in the image?', 
    images=['https://cdn-icons-png.flaticon.com/256/6028/6028690.png'],
    return_history=True,
    **gen_kwargs
)
print('response:')
print(response)
print('history:')
print(history)
# response: 
# The image contains a large, stylized "HI!" in a bright pink color with a yellow outline. The "HI!" is in a speech bubble shape.

# history: 
# [['What is in the image?', 'The image contains a large, stylized "HI!" in a bright pink color with a yellow outline. The "HI!" is in a speech bubble shape.']]

#################################################################################
# second round
#################################################################################
response, history = model.chat(
    texts='Are you sure?', 
    images=['https://cdn-icons-png.flaticon.com/256/6028/6028690.png'], # images need to be passed again in multi-round conversations
    history=history,
    return_history=True,
    **gen_kwargs
)

print('response:')
print(response)
print('history:')
print(history)
# response: 
# Yes, I'm sure. The image shows a large, stylized "HI!" in a bright pink color with a yellow outline, placed in a speech bubble shape.

# history: 
# [['What is in the image?', 'The image contains a large, stylized "HI!" in a bright pink color with a yellow outline. The "HI!" is in a speech bubble shape.'], ['Are you sure?', 'Yes, I\'m sure. The image shows a large, stylized "HI!" in a bright pink color with a yellow outline, placed in a speech bubble shape.']]
