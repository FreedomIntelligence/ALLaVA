

vflan_root="allava_vflan"


cd $vflan_root

# 1. download annotation files 
## 1.1 caption
wget -c -O ALLaVA-Caption-VFLAN-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_vflan/ALLaVA-Caption-VFLAN-4V.json?download=true

## 1.2 instruction
wget -c -O ALLaVA-Instruct-VFLAN-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_vflan/ALLaVA-Instruct-VFLAN-4V.json?download=true


# 2. download and upzip images
mkdir images
cd images

wget  -c -O "image_191-task_1k.zip" "https://huggingface.co/datasets/Vision-Flan/vision-flan_191-task_1k/resolve/main/image_191-task_1k.zip?download=true"

unzip image_191-task_1k.zip