Figure 5A ROI source data

This supplement contains only the displayed Cellpose3 and Cellpose-SAM region.
The original full-resolution H&E data have been provided separately.

Region: rows 3650-3849, columns 5100-5299 in the processed H&E image; zero-based coordinates from the upper-left corner. Every supplied image is a 200 x 200-pixel crop.

HE_ROI.tif: shared H&E background without overlays.
*_mask.tif: cropped segmentation labels, with their saved label values unchanged.
*_boundary.tif: boundaries extracted from the full label maps before cropping.
*_cells.csv: saved annotation points and cell types in the region. center_x is the local row; center_y is the local column.
celltype_colors.csv: the original cell-type palette and category order.

Original_results contains the archived source-panel PDFs and crops from the final assembled panels. The source plots use the original RGB plotting colors; the final panel crops retain the assembled figure appearance.

Run python reproduce_ROI.py from the package main folder. It creates two source-plot images in Reproduced, using the archived drawing settings.

ROI_review.ipynb provides a step-by-step interactive view of the same data and plots. Its executed outputs are saved for immediate inspection.

H&E provenance: the ROI was extracted from the preprocessed Xenium breast cancer H&E released with UCS.
Public data source: https://drive.google.com/file/d/1vhTHboLGF9jCb9vNuzHzaR8fU1Jg4qV5/view
