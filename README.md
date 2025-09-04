# eeg-fmri preproc

Tools for analysing EEG acquired during simultaneous EEG-fMRI experiments.  

This is an add-on package designed to be used with [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) and [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics), tailored for EEG-fMRI specific preprocessing and analysis.  

The package is currently in early **alpha** development. If you encounter any bugs, please report them via the *Issues* page.  

## Installation

We recommend installing this package in a **conda** environment.  

1. First install [osl-ephys](https://github.com/OHBA-analysis/osl-ephys) and [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics). You can install two different environments for osle and osld.
2. Then, inside the same environment, run:  

```
pip install torch torchvision
pip install natsort tqdm
pip install py-ecg-detectors
```

## Usage

For data from the Staresina Lab, the expected directory structure is:

```
/path/to/your/base/
├── eeg-fmri_Staresina/
│   ├── edfs/          # Uncompressed .edf and .edf.dpa files
│   ├── after_prep/    # Preprocessed data
│   ├── after_src/     # Source reconstructed data
│   ├── after_hmm/     # HMM results
│   └── sub-*          # raw data stored here. required for polhemus information
│
└── eeg-fmri-preproc/
    └── scripts/
        └── sts/       # Run scripts with: python *.py
```

The files in edfs requires following tuning:
- sub028-ses01-run02-block01 has multiple files. Retain only one.
- sub001 has a different style of naming. Rename it before running.

## TODO

1. py-ecg-detectors is not required. Copy the code and remove the requirement.
2. Folder structure just tuned, import error may exist. 
3. Multiple pathfinder exists. Make it generalizable and retain only one.
