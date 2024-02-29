

laion_root="allava_laion"

mkdir $laion_root
cd $laion_root


# 1. download annotation files 
## 1.1 caption
wget -c -O ALLaVA-Caption-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Caption-LAION-4V.json?download=true

## 1.2 instruction
wget -c -O ALLaVA-Instruct-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Instruct-LAION-4V.json?download=true


# 2. download and upzip images
wget -c -O images.zip https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/images.zip?download=true

unzip images.zip # wait patiently, it takes a while...


