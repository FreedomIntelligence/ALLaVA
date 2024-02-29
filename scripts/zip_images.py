
import json

from tqdm import tqdm


from PIL import Image
import os, pdb

import multiprocessing as mp




'''
id_set = set()
id2line = {}

for line in tqdm(lines):
    image = line['image']
    id = image.split('/')[-1]
    if id in id_set:
        pdb.set_trace()
    id_set.add(id)
    id2line[id] = line

print(len(lines))

print(len(id_set))
pdb.set_trace()
'''

'''
allava_laion/
    images/
    cap.json
    inst.json

'''

output_dir = '/mntcephfs/data/med/guimingchen/workspaces/vllm/upload/ALLaVA/dataset_v2'




def process_image(line):
    '''
    Function to process a single image
    '''
    # line['image'] = line['image'].replace('/mntcephfs/data/med/zhanghongbo/MOSS/cjy/cjy_data', '/wangbenyou/guimingchen/datasets/laion')
    img = Image.open(line['image'])
    img_format = img.format 
    int(line['id'])
    img_name = line['id'] + "." + img_format.lower()

    dst = os.path.join(output_dir, 'allava_laion/images', img_name)

    if not os.path.exists(dst):
        os.symlink(line['image'], dst)
    return dst


def process_images():
    '''
    create a soft link for each image
    '''

    with open('/mntcephfs/data/med/shunian/vlm/data/huggingface_version/laion_v3.json') as f:
        lines = json.load(f)[:]
    
    pdb.set_trace()

    # # create a dict mapping each image path to an int. The int will be the released id.
    # with open('/wangbenyou/guimingchen/workspaces/vllm/upload/hf/dataset_v2/path2id.json') as f:
    #     global path2id
    #     path2id = json.load(f)

    # number of processes to create
    process_num = mp.cpu_count()-2
    print(f'using {process_num}')

    with mp.Pool(process_num) as pool:
        # this uses tqdm for a progress bar
        list(tqdm(pool.imap(process_image, lines), total = len(lines)))

process_images()