#%%
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
# import osl_ephys as osle
import osl_dynamics as osld
# from osl_ephys.utils import Study
from osl_dynamics.data import Data
from osl_dynamics.analysis import modes, power
from osl_dynamics.models.hmm import Config, Model

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
sys.path.append(os.path.abspath(os.getcwd()))
from utils import Pathfinder
import pickle
from osl_dynamics.inference import tf_ops

# GPU settings
pf = Pathfinder(recon="after_src_sts", prepared="after_prepared_sts", hmm="after_hmm_sts")
pth = pf.get_fdr_dict()
tf_ops.gpu_growth()

# Get source data files
src_dir = pth['recon']
src_data_files = sorted(glob(src_dir + "/*/*sflip*raw.fif"))
training_data = Data(src_data_files)
methods = {
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
}
training_data.prepare(methods)
training_data.save(pth['prepared'])

# Settings
config = Config(
    n_states=8,
    n_channels=80,
    sequence_length=2000,
    learn_means=False,
    learn_covariances=True,
    learn_trans_prob=True,
    batch_size=64,
    learning_rate=0.01,
    n_epochs=25,
)

# Load training data
# training_data = Data(
#     "/well/woolrich/projects/lemon/chet22/prepared38",
#     store_dir=f"tmp_{run}",
# )

# Build model
model = Model(config)
model.summary()

# Initialization
init_history = model.random_state_time_course_initialization(training_data, n_init=3, n_epochs=3)

print("Training model")
history = model.fit(training_data)

# Save the trained model
model.save(f"{pth['hmm']}")

# Save training history
with open(f"{pth['hmm']}/history.pkl", "wb") as file:
    pickle.dump(history, file)

with open(f"{pth['hmm']}/loss.dat", "w") as file:
    file.write(f"loss = {history['loss'][-1]}\n")

# Get inferred covariances
covs = model.get_covariances()

# Plot inferred power maps
raw_covs = modes.raw_covariances(
    covs,
    n_embeddings=training_data.n_embeddings,
    pca_components=training_data.pca_components,
)
power.save(
    power_map=raw_covs,
    filename=f"{pth['hmm']}/covs_.png",
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    subtract_mean=True,
)

# Delete temporary directory
training_data.delete_dir()
