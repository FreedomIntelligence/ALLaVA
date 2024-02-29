import json
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import Pool


############### INPUT and OUTPUT path ###############
hf_laion_caption_path = '/path/to/ALLaVA-Caption-LAION-4V.json'
laion_caption_output_path = '/path/to/ALLaVA-Caption-LAION-4V_with_image.json'

hf_laion_inst_path = '/path/to/ALLaVA-Instruct-LAION-4V.json' # 
laion_inst_output_path = '/path/to/ALLaVA-Instruct-LAION-4V_with_image.json'

image_dir = '/path/to/image_dir'
############### INPUT and OUTPUT path ###############





def download_single_image(line):
    try:
        url = line['url']
        image_path = os.path.join(args.image_dir, f'allava_laion_{line["id"].split("_")[-1]}') 
        # allava_laion_0, allava_laion_1, allava_laion_2, ...
        # note that they are saved as binary files.
        # each file can be loaded with Image.open()

        if os.path.exists(image_path):
            line['image'] = image_path
            return line

        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            # save as a binary file
            with open(image_path, 'wb') as file:
                file.write(response.content)
            line['image'] = image_path
            return line
        else:
            return None
        
    except Exception as e:
        # remove the binary image file
        if os.path.exists(image_path):
            os.remove(image_path)
        return None



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', default='', type=str)

    parser.add_argument('--hf_laion_caption_path', required=True)
    parser.add_argument('--laion_caption_output_path', required=True)

    parser.add_argument('--hf_laion_inst_path', required=True)
    parser.add_argument('--laion_inst_output_path', required=True)

    parser.add_argument('--num_processes', default=200, type=int)

    args = parser.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)


    for input_path, output_path in (
        [args.hf_laion_caption_path, args.laion_caption_output_path], # this step takes long time to run. The code supports continual download so you can interupt and rerun at anytime.
        [args.hf_laion_inst_path, args.laion_inst_output_path] # this step takes little time to run since it shares the same set of images with caption
    ):

        with open(input_path) as f:
            data = json.load(f)

        with Pool(processes=args.num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(download_single_image, data), total=len(data)))

        # filter None
        results = [da for da in results if da is not None]

        print('downloaded image:', len(results))

        # save
        os.path.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as fw:
            json.dump(results, fw, ensure_ascii=False, indent=2)
