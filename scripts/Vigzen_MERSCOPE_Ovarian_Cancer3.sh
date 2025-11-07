Download Vizgen Mouse Brain S2R1 from https://console.cloud.google.com/storage/browser/public-datasets-vizgen-merfish/
# the data root,replace it with your own path

# Run the preprocess to get the gene map and vizgen segmentation
python preprocess/vizgen.py \
--transcripts C:/Users/lyh/segmentation_datasets/Vigzen_MERSCOPE_Ovarian_Cancer3/detected_transcripts_S2R1.csv \
--cell_boundaries C:/Users/lyh/segmentation_datasets/Vigzen_MERSCOPE_Ovarian_Cancer3/cell_boundaries/ \
--cell_meta C:/Users/lyh/segmentation_datasets/Vigzen_MERSCOPE_Ovarian_Cancer3/cell_metadata_S2R1.csv \
--transform_matrix C:/Users/lyh/segmentation_datasets/Vigzen_MERSCOPE_Ovarian_Cancer3/micron_to_mosaic_pixel_transform.csv \
--images C:/Users/lyh/segmentation_datasets/Vigzen_MERSCOPE_Ovarian_Cancer3/images/ \
--out_dir ./data/Vigzen_MERSCOPE_Ovarian_Cancer3

# You can also use the Cellpose Nuclei for SpaCut with the Maximum Intensity Projection (MIP) of DAPI channel
# reference cellpose_sam.ipynb to get the nuclei mask using Cellpose-SAM
# reference cellpose.ipynb to get the nuclei mask using Cellpose

# ========================================================================================================
# [Optional]Check the gene map and nuclei mask by visualizing the region [x_min, x_max, y_min, y_max]
python preprocess/check_paired.py \
--gene_map ./data/Vigzen_MERSCOPE_Ovarian_Cancer3/gene_map.tif \
--segmentation ./data/Vigzen_MERSCOPE_Ovarian_Cancer3/cell_vizgen_mask.tif \
--region 2000 3000 2000 3000 \
--out_dir ./data/Vigzen_MERSCOPE_Ovarian_Cancer3


# Run on Vigzen MERSCOPE Ovarian Cancer3 dataset
python run.py --gene_map ./data/Vigzen_MERSCOPE_Ovarian_Cancer3/gene_map.tif \
--nuclei_mask ./data/Vigzen_MERSCOPE_Ovarian_Cancer3/cell_vizgen_mask.tif \
--log_dir ./logs/Vigzen_MERSCOPE_Ovarian_Cancer3_cellmask \