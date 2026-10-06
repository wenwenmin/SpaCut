"""Recreate the two archived ROI source plots.
Run: python reproduce_ROI.py
Requires tifffile, pandas, matplotlib and seaborn.
"""
from pathlib import Path
import tifffile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'Source_data'
OUT=ROOT/'Reproduced'
OUT.mkdir(exist_ok=True)
he=tifffile.imread(DATA/'HE_ROI.tif')
palette=pd.read_csv(DATA/'celltype_colors.csv')

for method in ['Cellpose3','Cellpose-SAM']:
    mask=tifffile.imread(DATA/f'{method}_mask.tif')
    boundary=tifffile.imread(DATA/f'{method}_boundary.tif').astype(bool)
    assert he.shape==(200,200,3) and mask.shape==boundary.shape==(200,200)
    cells=pd.read_csv(DATA/f'{method}_cells.csv')
    cells['C_scANVI']=pd.Categorical(cells['C_scANVI'],
        categories=palette['cell_type'].tolist(),ordered=True)
    fig,ax=plt.subplots(figsize=(6,6))
    ax.imshow(he,aspect='auto')
    ax.imshow(boundary,cmap='Reds',alpha=0.6)
    sns.scatterplot(x=cells['center_y'],y=cells['center_x'],
        hue=cells['C_scANVI'],s=10,palette=palette['color_hex'].tolist(),
        edgecolors='none',ax=ax)
    plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    ax.set_axis_off()
    fig.savefig(OUT/f'{method}.png',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {method}.png')
