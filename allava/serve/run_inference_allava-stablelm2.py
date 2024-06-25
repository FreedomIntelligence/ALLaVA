from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pdb


dir = "FreedomIntelligence/ALLaVA-StableLM2-1_6B"

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(dir, trust_remote_code=True)
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
The image contains a graphic design of a speech bubble with the word "HI!" written inside it. The speech bubble is colored in pink and has yellow outlines.

history:
[['What is in the image?', 'The image contains a graphic design of a speech bubble with the word "HI!" written inside it. The speech bubble is colored in pink and has yellow outlines.']]
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
Yes, I am certain. The image displays a graphic design of a speech bubble with the word "HI!" written inside it. The speech bubble is colored in pink and has yellow outlines.

history:
[['What is in the image?', 'The image contains a graphic design of a speech bubble with the word "HI!" written inside it. The speech bubble is colored in pink and has yellow outlines.'], ['Are you sure?', 'Yes, I am certain. The image displays a graphic design of a speech bubble with the word "HI!" written inside it. The speech bubble is colored in pink and has yellow outlines.']]
'''