"""
This module initializes the utility package by importing all necessary submodules.

Submodules:
    base: Contains base utility extension functions and classes for python. e.g. io
    eeg: Contains utilities for EEG data processing.
    mri: Contains utilities for MRI data processing.
    matlab: Contains utilities for interfacing with MATLAB engine.

Usage:
    Importing this module will automatically import all functions and classes from the submodules. However, this may not be feasible as the package for specific submodules may not be installed. In such cases, import the specific submodule directly.
"""

from .util import ensure_dir
from .eeg import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, Pathfinder, filename2subj, HeteroStudy, pick_indices, plot_channel_dists, pcs_plot
from .qrs import QRSDetector
# from .mri import pad_img, unpad_img, read_data, write_data, get_excircle_square
# from .matlab import MatlabInstance