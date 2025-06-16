"""
This module initializes the utility package by importing all necessary submodules. Would lead to slower import time, but allows for easier debug and development.

Submodules:
    base: Contains base utility extension functions and classes for python. e.g. io
    eeg: Contains utilities for EEG data processing.
    mri: Contains utilities for MRI data processing.
    matlab: Contains utilities for interfacing with MATLAB engine.

Usage:
    Importing this module will automatically import all functions and classes from the submodules. However, this may not be feasible as the package for specific submodules may not be installed. In such cases, install the specific submodule.
"""

from .util import ensure_dir
from .eeg import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, Pathfinder, filename2subj, pick_indices, pcs_plot, HeteroStudy
from .qrs import QRSDetector
from .lemon_prep import lemon_init, LemonPathFinder
try:
    from .osle_expansion import plot_channel_dists
except:
    print("Warning: osl_ephys not found. Expansion based on osle would not be available.")
try:
    from .mri import pad_img, unpad_img, read_data, write_data, get_excircle_square
except:
    print("Warning: sitk not found. MRI utilities would not be available.")
# from .matlab import MatlabInstance