#!/bin/bash

# add help statement
if [[ $# -eq 0 || $1 == "-h" || $1 == "--help" ]]; then
    echo 'Usage: consolidate_gcp.bash <gcp_link>'
    exit 1
fi

# 1 arg = gcp link
gcp_link=$1

# eg: gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/vicuna-7b-DFN-CLIP-737k-bs512-res224

# if doesnt start with gcp, then its a local path
if [[ $gcp_link != gs* ]]; then
    echo "Not a gs link. Recieved: $gcp_link"
    exit 1
fi

# extract the filename from the last /
filename=$(echo $gcp_link | rev | cut -d'/' -f1 | rev)

# ensure prepended with "llava-" filename to work with evals
if [[ $filename != llava* ]]; then
    echo "Prepending llava- to $filename"
    filename="llava-$filename"
fi

# TODO: I think we need to create a new dir in which to svae the full model?

# 1. download the file
ckpt_path=./checkpoints/$filename
echo "Downloading from $gcp_link to $ckpt_path"
gcloud alpha storage rsync $gcp_link $ckpt_path

# done downloading
echo "Downloaded to $ckpt_path."


# 2. now run the consolidate script
echo "Consolidating $ckpt_path"
python consolidate.py --ckpt_path $ckpt_path
echo "Consolidated $ckpt_path"

# 3. convert_hf_model
echo "Converting $ckpt_path to HF model"
python convert_hf_model.py --ckpt_path $ckpt_path
echo "Converted $ckpt_path to HF model"

# # now, launch the evaluation script
# echo "Launching evaluation script for $ckpt_path"

# python launch_evaluation.py --model_path $ckpt_path


find_available_port() {
  local port=$(shuf -i 1025-65535 -n 1)
  while ss -ltn | awk '{print $4}' | grep -q ":$port$"; do
    port=$(shuf -i 1025-65535 -n 1)
  done
  echo "$port"
}
port=$(find_available_port)
echo "Randomly selected port: $port"

# 4. launch the evaluation script
echo "Launching evaluation script for $ckpt_path"
nohup python launch_evaluation.py --model_path $ckpt_path --port $port > eval_$filename.log 2>&1 &
