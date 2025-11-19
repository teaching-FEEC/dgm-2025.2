#!/bin/bash

# List of wandb run IDs
run_ids=(
    hxgjenmk
    lzh6cqwh

#   i5d529hc
#   rpxuujz1

#   w2yjc957
#   lvpv5vih
#   sib24cd6
#   y981mzsk
#   j6kdgk6h
#   fe1sh37q
#   hsuam3pg
#   q72x7u90
#   krokzi7r
)

# Base path for logs
base_dir="logs/hypersynth"

# Loop over all run IDs
for run_id in "${run_ids[@]}"; do
  config_path="$base_dir/$run_id/config.yaml"
  ckpt_dir="$base_dir/$run_id/checkpoints"

  # Search for checkpoint(s) that start with "epoch=" and end with ".ckpt"
  ckpt_path=$(find "$ckpt_dir" -type f -name "epoch=*.ckpt" | head -n 1)

  if [ -f "$ckpt_path" ]; then
    echo "Running validate for $run_id with checkpoint: $ckpt_path"
    python src/main.py validate -c "$config_path" --ckpt_path="$ckpt_path"
  else
    echo "Skipping $run_id — no checkpoint found in $ckpt_dir"
  fi
done