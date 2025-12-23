# Spectral Geometry of Fungal Interactions

This repository contains the Python source code used for the spectral geometric modeling and chemical decoding analysis presented in the manuscript: "Decoding the Spectral Geometry of Fungal Interactions: Machine Learning Reveals Latent Ecological Memory."

## Contents
* `tricho_ring_figures.py`: The main script that performs vector analysis on hyperspectral data to generate the geometric trajectory plots (Figure 9) and spectral residual decoding (Figure 10).
*`average_data.xlsx`: The processed input dataset containing mean spectral profiles extracted from the Regions of Interest (ROIs).

## Note on Machine Learning
The machine learning classification (Neural Boosted, Random Forest, etc.) described in the manuscript was performed using **JMP Pro 17** (SAS Institute). As JMP is a GUI-based software, no script files are available for that portion of the analysis. Full statistical details are provided in the manuscript Methods section.

## Usage
To reproduce the geometric analysis:
```bash
python tricho_ring_figures.py --excel "path/to/data.xlsx" --out "output_dir"
