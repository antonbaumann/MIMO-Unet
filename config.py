"""This is the configuration file for the training. Written in python to allow importing, typing and code execution."""
from pathlib import Path

# contains ROIs which are predicted each epoch to allow visual inspection of results of the model
# the ROIs are not included in the training set.
VALIDATION_INSPECTION_DIR = Path("/scratch/thomas/2021-09-02-inspection-rois-dataset/")
