# Project Title: Torch RDM Searchlight Analysis

## Overview
This project implements a Representational Dissimilarity Matrix (RDM) analysis using a searchlight approach with PyTorch. It reads fMRI beta images, computes RDMs based on specified conditions, and visualizes the results, including glass brain images and RDM plots.

## Project Structure
```
torch-rdm-searchlight
├── src
│   ├── main.py               # Entry point for the application
│   ├── rdm_utils.py          # RDM calculations using PyTorch
│   ├── visualization.py       # Visualization functions for results
│   ├── searchlight.py          # Searchlighting
│   └── types
│       └── index.py          # Custom types and data structures
├── data
│   └── firstlevel_forRSA
│       ├── beta_0002.nii     # Beta image for positive condition
│       ├── beta_0004.nii     # Beta image for negative condition
│       ├── beta_0006.nii     # Beta image for unrelated condition
│       └── mask.nii          # Mask file for data processing
├── outputs
│   ├── glass_brain.png       # Glass brain visualization output
│   ├── roi_results.csv        # ROI results in CSV format
│   ├── roi_rdm.png           # RDM visualization output
│   ├── roi_rdm.npy           # RDM data in NumPy format
│   └── roi_coordinates.txt    # Coordinates of the ROIs
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd torch-rdm-searchlight
pip install -r requirements.txt
```

## Usage
To run the analysis, execute the main script:

```bash
python src/main.py
```

This will read the beta files from the `data/firstlevel_forRSA` directory, compute the RDMs, and save the results in the `outputs` directory.

## Data
The project uses the following beta images:
- `beta_0002.nii`: Represents the positive condition.
- `beta_0004.nii`: Represents the negative condition.
- `beta_0006.nii`: Represents the unrelated condition.

A mask file (`mask.nii`) is also included to ensure proper data processing.

## Outputs
The results of the analysis will be saved in the `outputs` directory, including:
- `glass_brain.png`: Visualization of active brain regions.
- `roi_results.csv`: Results of the ROI analysis.
- `roi_rdm.png`: Visualization of the RDM with numerical values displayed.
- `roi_rdm.npy`: The RDM data in NumPy format.
- `roi_coordinates.txt`: Text file containing the coordinates of the regions of interest (ROIs).

## Coordinates of ROIs
The following MNI coordinates are used for the analysis:
- ROI 1: (15, 12, 66)
- ROI 2: (-66, -21, 15)
- ROI 3: (-48, 36, 24)
- ROI 4: (-24, -12, -24)

## License
This project is licensed under the MIT License - see the LICENSE file for details.
