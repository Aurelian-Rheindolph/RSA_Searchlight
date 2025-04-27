from typing import List, Tuple, Dict
import numpy as np

# Define a type for the conditions used in the RDM calculations
Condition = str

# Define a type for the RDM, which is a 2D numpy array
RDM = np.ndarray

# Define a type for the coordinates of the ROIs
ROI_Coordinates = List[Tuple[float, float, float]]

# Define a type for the output results, which includes the RDM and ROI coordinates
OutputResults = Dict[str, RDM]  # Keyed by result type (e.g., 'RDM', 'ROI')