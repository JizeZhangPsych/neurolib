#%% 
import re
import os, sys, mne, copy, glob
import os.path as op
from pprint import pprint
import osl_ephys
from osl_ephys import source_recon, utils as osl_utils
import numpy as np

continue_interrupt = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import psd_plot, temp_plot, temp_plot_diff, mne_epoch2raw, parse_subj, Pathfinder, filename2subj, HeteroStudy as Study

def polhemus_translation(outdir, subject):
    unused_ch = ['F11', 'F12', 'FT11', 'FT12', 'Cb1', 'Cb2']
    
    rhino_pth_dict = source_recon.rhino.get_coreg_filenames(outdir, subject)
    subj_dict = parse_subj(subject)

    polhemus_files = Study([
        os.path.join(os.path.dirname(outdir) + '/{subj}/polhemus/{subj}_{ses}_{run}_{foo}.pom'),
        os.path.join(os.path.dirname(outdir) + '/{subj}/{ses}/polhemus/{subj}_{ses}_{run}_{foo}.pom')
    ])
    polhemus = polhemus_files.get(subj=subj_dict["subj"], ses=subj_dict["ses"], block=subj_dict["block"], run=subj_dict["run"])
    assert len(polhemus) == 1, f"Expected one polhemus file, found {len(polhemus)} for subject {subject}."
    polhemus = polhemus[0]
    # Extract LOCATION_LIST data
    with open(polhemus, 'r') as f:
        polhemus_content = f.read()
    polhemus_content = re.sub(r"#.*?\n", "\n", polhemus_content)  # Remove comments

    location_list_match = re.search(r"LOCATION_LIST START_LIST([\s\S]*?)LOCATION_LIST END_LIST", polhemus_content)
    assert location_list_match is not None
    locations_data = location_list_match.group(1).strip().splitlines()
    locations = [line.split() for line in locations_data]
    locations = np.asarray(locations, dtype=np.float32)

    remark_list_match = re.search(r"REMARK_LIST START_LIST([\s\S]*?)REMARK_LIST END_LIST", polhemus_content)
    assert remark_list_match is not None
    remarks = remark_list_match.group(1).strip().splitlines()

    sign_x = (locations[remarks.index('C6')][0] > locations[remarks.index('C5')][0]) * 2 - 1
    sign_y = (locations[remarks.index('Fpz')][1] > locations[remarks.index('Oz')][1]) * 2 - 1
    sign_z = (locations[remarks.index('Cz')][2] > np.mean(locations[:,2])) * 2 - 1
    locations = [[sign_x*x,sign_y*y,sign_z*z] for x,y,z in locations] 

    if locations[remarks.index('Left ear')][0] > locations[remarks.index('Right ear')][0]:
        locations[remarks.index('Left ear')][0] *= -1
        locations[remarks.index('Right ear')][0] *= -1
        locations[remarks.index('Nasion')][0] *= -1
    if locations[remarks.index('Nasion')][1] < (locations[remarks.index('Left ear')][1] + locations[remarks.index('Right ear')][1]) / 2:
        locations[remarks.index('Nasion')][1] *= -1
        locations[remarks.index('Left ear')][1] *= -1
        locations[remarks.index('Right ear')][1] *= -1

    # Iterate through remarks and locations
    headshape_coords = []

    for idx, remark in enumerate(remarks):
        if remark in unused_ch:
            continue
        
        if remark == 'Left ear':
            with open(rhino_pth_dict['polhemus_lpa_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        elif remark == 'Right ear':
            with open(rhino_pth_dict['polhemus_rpa_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        elif remark == 'Nasion':
            with open(rhino_pth_dict['polhemus_nasion_file'], 'w') as f:
                f.write(f"{locations[idx][0]}\n{locations[idx][1]}\n{locations[idx][2]}\n")
        else:
            headshape_coords.append(locations[idx])

    # Write headshape coordinates to file
    headshape_coords = np.array(headshape_coords).T  # Transpose to get x, y, z in separate rows
    with open(rhino_pth_dict['polhemus_headshape_file'], 'w') as f:
        for row in headshape_coords:
            row = [f"{coord:.6f}" for coord in row]  # Format to 6 decimal places
            f.write(' '.join(row) + '\n')
            
# Configure pipeline
        # - extract_polhemus_from_info:
            # include_eeg_as_headshape: true
config = """
    source_recon:
        - polhemus_translation: {}
        - compute_surfaces:
            include_nose: true
        - coregister:
            use_nose: true
            use_headshape: true
            allow_smri_scaling: true
        - forward_model:
            model: Triple Layer
            eeg: true
        - beamform_and_parcellate:
            freq_range: [1, 35]
            chantypes: eeg
            rank: {eeg: 45}
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
            reg: 0.05
"""


#%%
if __name__ == "__main__":
    no_polhemus_list = ['17111', '17112', '1121', '2121']
    pf = Pathfinder(prep="after_prep_sts", recon="after_src_sts")
    pth = pf.get_fdr_dict()
    osl_utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(pth['fsl'])
    
    ## eeg
    preproc_files = Study(os.path.join(pth["prep"], "{subject}/{subject}_preproc-raw.fif"))

    # other modalities

    smri_study = Study([
        pth["base"] + '{subj}/mri/{run}/{foo}/{foo1}t1{foo2}.nii',
        pth["base"] + '{subj}/{ses}/mri/{run}/{foo}/{foo1}t1{foo2}.nii'
    ])
    subject_list = [subject_string for subject_string in os.listdir(pth["prep"]) if subject_string.isdigit()]

    preproc_file_list = []
    smri_files = []
    
    skip_list = []
    for subject in subject_list:
        if subject in no_polhemus_list:
            print(f"WARNING: {subject} has no polhemus file, skipping")
            skip_list.append(subject)
            continue
        
        subj_dict = parse_subj(subject)
        
        # smri = smri_study.get(run=subj_dict["run"], subj=subj_dict["subj"], ses=subj_dict["ses"], block=subj_dict["block"])
        smri = smri_study.get(run=subj_dict["run"], subj=subj_dict["subj"], ses='ses-01', block=subj_dict["block"])        
        if not (len(smri)==2 or len(smri)==1):
            print(f"WARNING: {subject} has {len(smri)} smri files, skipping")
            skip_list.append(subject)
            continue
        
        preproc = preproc_files.get(subject=subject)
        if not len(preproc)==1:
            print(f"WARNING: {subject} has {len(preproc)} preproc files, skipping")
            skip_list.append(subject)
            continue

        smri_files.append(sorted(smri)[-1]) # only use normed one, assumed normed one always being the second. TODO ensure the assumption
        preproc_file_list.append(preproc[0])
        
    subject_list = [subj for subj in subject_list if subj not in skip_list]
    if continue_interrupt:
        full_preproc_file_list = preproc_file_list.copy()
        preproc_file_list = []
        full_subject_list = subject_list.copy()
        subject_list = []
        full_smri_file_list = smri_files.copy()
        smri_files = []
        
        # Check if files already processed
        finished_list = glob.glob(f'{pth["recon"]}/*/parc/lcmv-parc-raw.fif')
        finished_list = [full_string.split('/')[-3] for full_string in finished_list]
        error_list = glob.glob(f'{pth["recon"]}/logs/*.error.log')
        error_list = [full_string.split('/')[-1].split('_')[0] for full_string in error_list]
        for subject, preproc, smri in zip(full_subject_list, full_preproc_file_list, full_smri_file_list):
            if subject in finished_list:
                print(f"WARNING: {subject} already finished, skipping")
            elif subject in error_list:
                print(f"WARNING: {subject} had an error, skipping")
            else:
                preproc_file_list.append(preproc)
                subject_list.append(subject)
                smri_files.append(smri)
    # Initiate source reconstruction
    source_recon.run_src_batch(
        config,
        outdir=pth["recon"],
        subjects=subject_list,
        preproc_files=preproc_file_list,
        smri_files=smri_files,
        extra_funcs=[polhemus_translation],
        # dask_client=True,
    )

    
# %%
