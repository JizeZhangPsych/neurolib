#%% 
import re
import os, sys, glob, yaml
import os.path as op
from pprint import pprint
import osl_ephys
from osl_ephys import source_recon, utils as osl_utils
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, Pathfinder, filename2subj, HeteroStudy as Study

# SIGN FLIP DATA
if __name__ == "__main__":
    # Set logger
    pf = Pathfinder(recon="after_src_sts")
    pth = pf.get_fdr_dict()
    osl_utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(pth['fsl'])

    # Get subject IDs
    subject_ids = [
        file.split("/")[-3] for file in sorted(glob.glob(os.path.join(pth['recon'], "*/parc/*parc-raw.fif")))
    ]

    # Find a good template subject to align other subjects to
    template = source_recon.find_template_subject(
        pth['recon'], subject_ids, n_embeddings=15, standardize=True
    )
    print(f"Number of available subjects: {len(subject_ids)}")

    # Configure pipeline

    config_dict = {
        "source_recon": [
            {
                "fix_sign_ambiguity": {
                    "template": template,
                    "n_embeddings": 15,
                    "standardize": True,
                    "n_init": 3,
                    "n_iter": 5000,
                    "max_flips": 20
                }
            }
        ]
    }

    config = yaml.dump(config_dict)

    # Set up parallel processing
    # client = Client(n_workers=16, threads_per_worker=1)

    # Initiate sign flipping
    source_recon.run_src_batch(
        config,
        outdir=pth['recon'],
        subjects=subject_ids,
    )

    print("Sign flipping complete.")
