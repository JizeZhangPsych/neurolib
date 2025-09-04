import re
from pathlib import Path
from typing import Dict, Optional

from .base import BasePathfinder

class IrenePathfinder(BasePathfinder):
    """
    Concrete Pathfinder for the Irene EEG-fMRI dataset.
    
    File ID convention: concatenation of left stripped subject + session + run + block
    e.g., s02 visit1 block1 → "0211"
    """
    
    DEFAULT_FILE_PATTERNS = {
        'irene_ga': "/ohba/pi/knobre/irene/data_for_jize/curry_clean/visit{visit}/s{subject}/s{foo}_mrEEG{foo1}_block{block}_mr_clean.cdt",
        'rest': "/ohba/pi/knobre/irene/data_for_jize/raw/visit{visit}/s{subject}/s{foo}_mrEEG{foo1}_block{block}[._]{foo2}cdt",
        'irene_bcg': "/ohba/pi/knobre/irene/data_for_jize/curry_clean/visit{visit}/s{subject}/s{foo}_mrEEG{foo1}_block{block}_mr_bcg_clean.cdt",
        'preproc': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_irene/{subject}{visit}{block}/{subject}{visit}{block}_preproc-raw.fif",
        'preproc_wo_ica': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_prep_irene/{subject}{visit}{block}/{subject}{visit}{block}_raw_before_ica.pkl",
        'src': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_src_irene/{subject}{visit}{block}/parc/lcmv-parc-raw.fif",
        'irene_src': "/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina/after_src_ireneprep/{subject}{visit}{block}/parc/lcmv-parc-raw.fif"
    }

    def __init__(self, file_patterns: Optional[Dict[str, str]] = None):
        patterns = file_patterns if file_patterns is not None else self.DEFAULT_FILE_PATTERNS
        super().__init__(file_patterns=patterns)

    def dict2id(self, fields: Dict[str, Optional[str]]) -> str:
        """Convert fields dict to canonical file_id (numeric, no leading zeros)."""
        subject = fields.get('subject', "02")
        visit = fields.get('visit', "1")
        block = fields.get('block', "1")

        return f"{subject}{visit}{block}"

    def id2dict(self, file_id: str) -> Dict[str, str]:
        """Convert numeric file_id back to dict with stripped numbers."""
        # Adjust as needed depending on dataset
        if len(file_id) < 4:
            raise ValueError(f"Invalid Irene file_id: {file_id}")

        # Naive split: first chars for subj, then 1 digit each for ses, run, block
        # We can also assume subject always 1-3 digits, take all but last 3 digits
        subj_num = file_id[:2]
        visit_num = file_id[2]
        block_num = file_id[3]

        return {
            "subject": subj_num,
            "visit": visit_num,
            "block": block_num
        }
