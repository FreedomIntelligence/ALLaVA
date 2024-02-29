

################################################################################
hf_cap_ann_path="" # path to store hf caption annotation file
hf_inst_ann_path="" # path to store hf instruction annotation file

image_dir="" # directory to store images
cap_ann_with_image_path="" # path to store new *caption* annotation files with local image path
inst_ann_with_image_path="" # path to store new *instruction* annotation files with local image path
################################################################################


# 0. check file path
if [ "$hf_cap_ann_path" = "$cap_ann_with_image_path" ]; then
  echo "Input and output path are equal, exiting..."
  return 1 2>/dev/null
fi

if [ "$hf_inst_ann_path" = "$inst_ann_with_image_path" ]; then
  echo "Input and output path are equal, exiting..."
  return 1 2>/dev/null
fi


# 1. download annotation files from huggingface
## 1.1 caption
wget -c -O $hf_cap_ann_path https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/ALLaVA-Caption-LAION-4V.json?download=true

## 1.2 instruction
wget -c -O $hf_inst_ann_path https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/ALLaVA-Instruct-LAION-4V.json?download=true


# 2. download images from url
python ./download/laion/download_images_from_url.py \
    --hf_laion_caption_path $hf_cap_ann_path \
    --laion_caption_output_path $cap_ann_with_image_path \
    --hf_laion_inst_path $hf_inst_ann_path \
    --laion_inst_output_path $inst_ann_with_image_path \
    --image_dir $image_dir \
    --num_processes 200

