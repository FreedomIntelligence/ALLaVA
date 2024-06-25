from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pdb

dir = "FreedomIntelligence/ALLaVA-Phi2-2_7B"

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(dir)
model.tokenizer = tokenizer

gen_kwargs = {
    'min_new_tokens': 20,
    'max_new_tokens': 100,
    'do_sample': False,
    'eos_token_id': tokenizer.eos_token_id,
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
print()
print('history:')
print(history)

'''
response:
The image contains a large, stylized "HI!" in a bright pink color with yellow outlines. The "HI!" is placed within a speech bubble shape.

history:
[['What is in the image?', 'The image contains a large, stylized "HI!" in a bright pink color with yellow outlines. The "HI!" is placed within a speech bubble shape.']]
'''


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
print()
print('history:')
print(history)

'''
response:
Yes, I'm certain. The image is a graphic representation of the word "HI!" in a speech bubble.

history:
[['What is in the image?', 'The image contains a large, stylized "HI!" in a bright pink color with yellow outlines. The "HI!" is placed within a speech bubble shape.'], ['Are you sure?', 'Yes, I\'m certain. The image is a graphic representation of the word "HI!" in a speech bubble.']]
'''