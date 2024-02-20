from allava.constants import IMAGE_TOKEN_INDEX
from allava.model import *

from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import torch

from PIL import Image
import pdb

KEYWORDS_IN_PATH = ['allava-3b', 'allava-3b-longer', 'phi']


class Chatbot():
    def __init__(self, config):
        self.config = config


        self.gen_kwargs = {
            'do_sample': False,
            'max_new_tokens': 768,
            'min_new_tokens': 1,
        }

        self.device = getattr(config, 'device', 'cuda')
        self.init_components()

        self.history = []
        self.images = []

        # although we support multiple image inputs at inference, this feature is NOT trained. Therefore, inputing multiple images may cause a degraded model performance.
        self.max_images_per_round = getattr(config, 'max_images_per_round', 3)

    def init_components(self):
        d = self.config.model_dir


        if any([name in d.lower() for name in KEYWORDS_IN_PATH]):
            print(f'loading from {self.config.model_dir}')
            model, loading_info = LlavaPhiForCausalLM.from_pretrained(self.config.model_dir, init_vision_encoder_from_ckpt=True, output_loading_info=True, trust_remote_code=True)

            missing_keys = loading_info['missing_keys'] # keys exists in model architecture but does not exist in ckpt
            unexpected_keys = loading_info['unexpected_keys'] # keys exists in ckpt but are not loaded by the model 
            assert missing_keys == [] and unexpected_keys == [] # both should be empty

            self.maxlen = getattr(self.config, 'maxlen', model.config.max_position_embeddings)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_dir, model_max_length=self.maxlen, trust_remote_code=True)
            vision_tower = model.get_vision_tower()

            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=self.device).half()
            image_processor = vision_tower.image_processor
            eos_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self.gen_kwargs['eos_token_id'] = tokenizer.eos_token_id # new features in transformers 4.37, where you need to explicitly pass the eos_token_id as a param
            self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
            print(f'setting eos_token_id to {eos_token_id}')


        else:
            print(f'please load your model properly.')
            raise NotImplementedError

        model.eval()
        self.model = model.half().to(self.device)
        self.tokenizer = tokenizer
        self.processor = image_processor


    def clear_history(self,):
        self.images = []
        self.history = []
        self.model.cached_image_features = None


    # copied from llava
    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): 
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


    def preprocess(self, data: list, return_tensors='pt'):
        '''
        [
            {
                'from': 'human',
                'value': xxx,
            },
            {
                'from': 'gpt',
                'value': xxx
            }
        ]
        '''
        # needs update
        if not isinstance(data, list):
            raise ValueError('must be a list')
        
        d = self.config.model_dir
        
        # this is per model (tokenizer)
        if any([name in d.lower() for name in KEYWORDS_IN_PATH]):
            return self.preprocess_allava(data, return_tensors=return_tensors)

        elif d in ['/path/to/llava-v1.5-13b']:
            return self.preprocess_vicuna_v1(data, return_tensors=return_tensors)
        
        else:
            raise NotImplementedError
        
        

    def preprocess_vicuna_v1(self, convs: list, return_tensors) -> list: # tokenize and concat the coversations
        input_ids = None
        for ind, conv in enumerate(convs):
            if ind % 2 == 0: # human
                h = conv['value'].strip()
                h = f"USER: {h} " 
                cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)
                
                if input_ids is None:
                    input_ids = cur_input_ids
                else:
                    input_ids = torch.cat([input_ids, cur_input_ids])

            else: # gpt
                g = conv['value']
                if g is not None:
                    cur_input_ids = self.tokenizer(f"ASSISTANT: {g}</s>", add_special_tokens= False, max_length=self.maxlen, truncation=True, return_tensors='pt').input_ids[0]
                    input_ids = torch.cat([input_ids, cur_input_ids])
                else:
                    cur_input_ids = self.tokenizer(f"ASSISTANT:", add_special_tokens= False, max_length=self.maxlen, truncation=True, return_tensors='pt').input_ids[0]
                    input_ids = torch.cat([input_ids, cur_input_ids])


        return input_ids

    def preprocess_allava(self, convs: list, return_tensors) -> list: # tokenize and concat the coversations
        input_ids = None

        for ind, conv in enumerate(convs):
            if ind % 2 == 0: # human
                h = conv['value'].strip()
                h = f"[INST] {h} [/INST] "
                cur_input_ids = self.tokenizer_image_token(prompt=h, return_tensors=return_tensors)
     
                if input_ids is None:
                    input_ids = cur_input_ids
                else:
                    input_ids = torch.cat([input_ids, cur_input_ids])

            else: # gpt
                g = conv['value']
                if g is not None:
                    cur_input_ids = self.tokenizer(f"{g}{self.tokenizer.eos_token}", add_special_tokens= False, max_length=self.maxlen, truncation=True, return_tensors='pt').input_ids[0]
                    input_ids = torch.cat([input_ids, cur_input_ids])

        return input_ids


    def input_moderation(self, t: str):

        blacklist = ['<image>', '<s>', '</s>']
        for b in blacklist:
            t = t.replace(b, '')
        return t
    
    def insert_image_placeholder(self, t, num_images, placeholder='<image>', sep='\n'):
        for _ in range(num_images):
            t = f"{placeholder}{sep}" + t

        return t
    
    def get_conv(self, text):
        ret = []
        if self.history is None:
            self.history = []
        
        for conv in self.history:
            ret.append({'from': 'human', 'value': conv[0]})
            ret.append({'from': 'gpt', 'value': conv[1]})

        ret.append({'from': 'human', 'value': text})
        ret.append({'from': 'gpt', 'value': None})
        return ret
    
    # copied from llava
    def get_image_tensors(self, images):
        list_image_tensors = []
        crop_size = self.processor.crop_size
        processor = self.processor
        for fp in images:
            if fp is None: # None is used as a placeholder
                list_image_tensors.append(torch.zeros(3, crop_size['height'], crop_size['width']).to(self.device))
                continue
            elif isinstance(fp, str):
                image = Image.open(fp).convert('RGB')
            elif isinstance(fp, Image.Image):
                image = fp # already an image
            else:
                raise TypeError(f'Unsupported type {type(fp)}')

            # this is the way of preprocessing images we used in training, so we impose it here
            if True:
                # self.data_args.image_aspect_ratio == 'pad'
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if pil_img.mode == 'L':
                        pil_img = pil_img.convert('RGB')

                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0] # a tensor
            list_image_tensors.append(image.to(self.device))
        return list_image_tensors


    def chat(self, text: str, images: list[str]=None, ):
        '''
        images: list[str], images for the *current* round
        text: text input for the *current* round
        '''

        ############################
        # 1. preprocess texts
        ############################
        text = self.input_moderation(text)
        if text == '':
            return 'Please type in something'

        if isinstance(images, str) or isinstance(images, Image.Image):
            images = [images]
        

        ############################
        # 2. preprocess images
        ############################
        valid_images = []
        if images is None:
            images = [None]
        
        for img in images:
            try:
                if isinstance(img, str):
                    Image.open(img).convert('RGB') # make sure that the path exists
                valid_images.append(img)
            except:
                continue
            
        images = valid_images

        if images == []  and self.images == []:
            self.images = [None]
            
        self.images.extend(images)

        assert len(images) < self.max_images_per_round, f'at most {self.max_images_per_round} images'

        ############################
        # 3. collate conv
        ############################

        # insert <image>
        text = self.insert_image_placeholder(text, len(images) if None not in images else 0)

        # collate strings into conv
        conv = self.get_conv(text)

        # make input ids
        input_ids = self.preprocess(conv, return_tensors='pt').unsqueeze(0).to(self.device)

        list_image_tensors = self.get_image_tensors(self.images)
        image_tensors = torch.stack(list_image_tensors)

        try:
            dtype = torch.bfloat16
            # if your hardware does not support bf16, the following line raises an error
            torch.tensor(1, dtype=dtype).cuda()
        except:
            # default using fp16
            dtype = torch.float16

        ############################
        # 4. generate response
        ############################
        with torch.autocast(device_type='cuda', dtype=dtype):
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_tensors,
                use_cache=getattr(self, 'use_cache', True),
                **self.gen_kwargs)

        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        # update history
        self.history.append([text, answer])
        return answer




if __name__ =="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Args of Data Preprocess')

    # Model Args
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--max_images_per_round', default=4, type=int)
    parser.add_argument('--maxlen', default=3500, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--stream',  action='store_true')
    args = parser.parse_args()

    bot = Chatbot(args)

    image_prompt = 'image pth, (split by "," for multiple images): '

    images = input(image_prompt)
    images = [i.strip() for i in images.split(',')]
    while True:
        text = input('USER ("clear" to clear history, "q" to exit): ')
        if text.lower() in ['q', 'quit']:
            exit()
        if text.lower() == 'clear':
            bot.clear_history()
            images = input(image_prompt)
            images = [i.strip() for i in images.split(',')]
            continue
        answer = bot.chat(images=images, text=text)
        images = None # already in the history
        print()
        print(f'GPT: {answer}')
        print()