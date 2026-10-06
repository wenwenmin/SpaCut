# Figure 5A materials

ROI data and plotting materials for the Cellpose3 and Cellpose-SAM panels in Figure 5A of SpaCut.

## Start here

Open [ROI_review.ipynb](ROI_review.ipynb) to inspect the shared H&E crop, the two reproduced source plots and the preserved final panels. Executed outputs are saved in the notebook for immediate viewing.

The supplied region is 200 x 200 pixels: rows 3650-3849 and columns 5100-5299 in the processed H&E image, using zero-based coordinates from the upper-left corner. Both methods use the same H&E background.

## Contents

| Item | Contents |
|---|---|
| `Source_data/` | H&E crop, segmentation labels, boundaries, annotation coordinates and cell-type colors for the two methods |
| `Original_results/` | Archived source-panel PDFs and preserved crops from the final assembled figure |
| `Reproduced/` | The two plots generated from the supplied ROI data |
| `reproduce_ROI.py` | Short script that recreates the source plots |
| `ROI_review.ipynb` | Step-by-step notebook with saved figure outputs |
| `Figure5A.pdf` | Final assembled Figure 5A for context |

## Reproduce the plots

From this folder, using Python 3.10 (the tested plotting environment):

```bash
pip install -r requirements.txt
python reproduce_ROI.py
```

The two images are saved in `Reproduced/`. To run the notebook interactively, open it in JupyterLab or another Jupyter-compatible editor using the same Python environment.

The script redraws saved segmentation and annotation outputs. Label values and annotation coordinates are preserved in the ROI extracts. Boundary maps were computed on the complete label maps before cropping. In the annotation CSVs, `center_x` is the local row and `center_y` is the local column.

The reproduced plots use the archived RGB plotting colors. The final panel crops preserve the assembled figure's display appearance.

## H&E source

The shared crop was extracted from the preprocessed Xenium breast cancer H&E distributed with [UCS](https://github.com/YangLabHKUST/UCS). The supplied preprocessed H&E was verified against the file in the [public downstream data package](https://drive.google.com/file/d/1vhTHboLGF9jCb9vNuzHzaR8fU1Jg4qV5/view).

This folder contains the two displayed ROIs; complete full-resolution H&E images and full segmentation or annotation datasets are not duplicated here.
