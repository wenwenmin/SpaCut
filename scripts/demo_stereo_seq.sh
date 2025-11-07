#!/bin/bash

# Demo: Stereo-seq Mouse Brain Dataset
# This script runs SpaCut on the preprocessed demo data

python run.py \
--gene_map demo_data/top500/top500_gene_map.tif \
--nuclei_mask demo_data/segmentation/nuclei_masks.tif \
--log_dir demo_data/spacut_results \
--patch_size 96 \
--dilation_kernel_size 15 \
--dilation_iter_num 5 \
--fg_net_epoch 10 \
--fg_net_batch_size 32 \
--fg_net_nuclei_weight 2 \
--cell_net_epoch 10 \
--cell_net_nuclei_weight 2 \