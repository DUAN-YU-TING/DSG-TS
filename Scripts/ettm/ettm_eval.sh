#!/bin/bash
python evaluation.py \
  --true_path The path to your ETTm1_24_norm_truth_24_test.npy\
  --gen_path The path to your ddpm_fake_ettm_24.npy \
  --gen_root Your path \
  --gen_name ddpm_fake_ettm_24.npy \
  --save_dir The save path for your own evaluation results for ettm_24 \
  --threshold 0.5

# python evaluation.py \
#   --true_path The path to your ETTm1_48_norm_truth_48_test.npy\
#   --gen_path The path to your ddpm_fake_ettm_48.npy \
#   --gen_root Your path \
#   --gen_name ddpm_fake_ettm_48.npy \
#   --save_dir The save path for your own evaluation results for ettm_48 \
#   --threshold 0.5

# python evaluation.py \
#   --true_path The path to your ETTm1_96_norm_truth_96_test.npy\
#   --gen_path The path to your ddpm_fake_ettm_96.npy \
#   --gen_root Your path \
#   --gen_name ddpm_fake_ettm_96.npy \
#   --save_dir The save path for your own evaluation results for ettm_96 \
#   --threshold 0.5

