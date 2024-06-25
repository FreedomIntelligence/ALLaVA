from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import pdb

dir = "FreedomIntelligence/ALLaVA-Phi3-mini-128k"

device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, device_map=device, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(dir)
model.tokenizer = tokenizer

gen_kwargs = {
    'min_new_tokens': 20,
    'max_new_tokens': 100,
    'do_sample': False,
    # eos_token_id is not needed for this model
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
- There is a speech bubble in the image.
- The speech bubble contains the word "HI!" in bold, yellow letters.

history:
[['What is in the image?', '- There is a speech bubble in the image.\n- The speech bubble contains the word "HI!" in bold, yellow letters.']]
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
- Yes, I am certain. The image prominently features a speech bubble with the word "HI!" inside it.

history:
[['What is in the image?', '- There is a speech bubble in the image.\n- The speech bubble contains the word "HI!" in bold, yellow letters.'], ['Are you sure?', '- Yes, I am certain. The image prominently features a speech bubble with the word "HI!" inside it.']]
'''