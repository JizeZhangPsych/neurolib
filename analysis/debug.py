#%%
import os, sys
sys.path.append('/ohba/pi/mwoolrich/jzhang/')
sys.path.append(os.path.abspath(os.getcwd()))
from analysis.obs import OBSAnalyser


#%%
obs_analyser = OBSAnalyser("3111", "after_prep1ai")
obs_analyser.print_pcs("tr_ep", "./foobar/pcs")
obs_analyser.print_noise_components("tr_ep", "./foobar/noise")
# %%
