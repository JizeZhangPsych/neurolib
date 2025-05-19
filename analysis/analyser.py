import os, pickle
from pathlib import Path
import mne
from functools import cached_property

class Analyser:
    def __init__(self, subject, mne_fdr, base_path="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina"):
        self.subject = subject
        self.path_dict = {
            "raw": os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}_ckpt_raw.pkl"),
            "output": os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}_preproc-raw.fif"),
        }
        try:
            with open(os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}_pf.pkl"), 'rb') as f:
                self.pf = pickle.load(f)
                
            assert Path(base_path).resolve() == Path(self.pf.fdr["base"]).resolve(), f"Base path {base_path} does not match the one in pf {self.pf.fdr['base']}"
            assert os.path.join(base_path, mne_fdr) == self.pf.fdr["mne"]
        except ModuleNotFoundError:
            pass

    @cached_property
    def raw(self):
        raw_pth = self.path_dict["raw"]
        with open(raw_pth, 'rb') as f:
            raw = pickle.load(f)
        return raw
    
    @cached_property
    def preproc(self):
        preproc_pth = self.path_dict["output"]
        preproc = mne.io.read_raw_fif(preproc_pth)
        return preproc
