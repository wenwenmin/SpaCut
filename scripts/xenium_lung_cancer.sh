
python preprocess/xenium.py \
--transcripts "D:/segmentation_datasets/Xenium_Human_Lung_Cancer/Xenium_V1_humanLung_Cancer_FFPE_outs/transcripts.csv" \
--cell_boundary_10X "D:/segmentation_datasets/Xenium_Human_Lung_Cancer/Xenium_V1_humanLung_Cancer_FFPE_outs/cell_boundaries.parquet" \
--nucleus_boundary_10X "D:/segmentation_datasets/Xenium_Human_Lung_Cancer/Xenium_V1_humanLung_Cancer_FFPE_outs/nucleus_boundaries.parquet" \
--out_dir ./data/xenium_human_lung_cancer

# [Optional]Check the gene map and nuclei mask by visualizing the region [x_min, x_max, y_min, y_max]
python preprocess/check_paired.py \
--gene_map ./data/xenium_human_lung_cancer/gene_map.tif \
--segmentation ./data/xenium_human_lung_cancer/nuclei_10X_mask.tif \
--region 2000 3000 2000 3000 \
--out_dir ./data/xenium_human_lung_cancer

# Run on Xenium Lung Cancer dataset
python run.py --gene_map ./data/xenium_human_lung_cancer/gene_map.tif \
--nuclei_mask ./data/xenium_human_lung_cancer/nuclei_10X_mask.tif \
--log_dir ./logs/xenium_human_lung_cancer