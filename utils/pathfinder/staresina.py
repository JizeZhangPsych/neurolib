import re
from pathlib import Path
from typing import Dict, Optional

from .base import BasePathfinder

class StaresinaPathfinder(BasePathfinder):
    """
    Concrete Pathfinder for the Staresina EEG-fMRI dataset.
    
    File ID convention: concatenation of left stripped subject + session + run + block
    e.g., sub-001 ses-01 run-01 block-02 → "1112"
    """
    
    DEFAULT_FILE_PATTERNS = {
        'a': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-0{subject}_ses-0{session}_run-0{run}_block-0{block}_task-resting_convert.cdt.edf",
        'rest': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/edfs/sub-0{subject}_ses-0{session}_run-0{run}_block-0{block}_task-resting_convert.cdt.edf",
        'polhemus': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/sub-0{subject}/ses-0{session}/polhemus/sub-0{subject}_ses-0{session}_run-0{run}_{foo}.opm",
        'preproc': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_sts/{subject}{session}{run}{block}/{subject}{session}{run}{block}_preproc-raw.fif",
        'src': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_src_sts/{subject}{session}{run}{block}/parc/lcmv-parc-raw.fif",
    }

    def __init__(self, file_patterns: Optional[Dict[str, str]] = None):
        patterns = file_patterns if file_patterns is not None else self.DEFAULT_FILE_PATTERNS
        super().__init__(file_patterns=patterns)

    def dict2id(self, fields: Dict[str, Optional[str]]) -> str:
        """Convert fields dict to canonical file_id (numeric, no leading zeros)."""
        subj = fields.get('subject')
        ses = fields.get('session', "1")
        run = fields.get('run', "1")
        block = fields.get('block', "1")

        # strip prefixes and leading zeros
        subj_num = subj.lstrip("0") if subj else "1"

        return f"{subj_num}{ses}{run}{block}"

    def id2dict(self, file_id: str) -> Dict[str, str]:
        """Convert numeric file_id back to dict with stripped numbers."""
        # assume file_id is numeric string like '1213' → sub=1, ses=2, run=1, block=3
        # length-based splitting: sub (1-3), ses (1), run (1), block (1)
        # Here we assume subject is first N chars, rest are single digits
        # Adjust as needed depending on dataset
        if len(file_id) < 4:
            raise ValueError(f"Invalid Staresina file_id: {file_id}")

        # Naive split: first chars for subj, then 1 digit each for ses, run, block
        # We can also assume subject always 1-3 digits, take all but last 3 digits
        subj_num = file_id[:-3]
        ses_num = file_id[-3]
        run_num = file_id[-2]
        block_num = file_id[-1]

        return {
            "subject": subj_num,
            "session": ses_num,
            "run": run_num,
            "block": block_num
        }
