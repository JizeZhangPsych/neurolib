import os, pickle, copy, sys
from .analyser import Analyser
from utils import pcs_plot, psd_plot, temp_plot, pick_indices, ensure_dir, mne_epoch2raw
import numpy as np

class OBSAnalyser(Analyser):
    def __init__(self, subject, mne_fdr, base_path="/ohba/pi/mwoolrich/datasets/eeg-fmri_Staresina"):
        super().__init__(subject, mne_fdr, base_path)
        self.path_dict["epoch"] = os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}" + "_{epoch_key}.pkl")
        self.path_dict["epoch_pc"] = os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}" + "_pc_{epoch_key}.pkl")
        self.path_dict["epoch_noise"] = os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}" + "_noise_{epoch_key}.pkl")
        self.path_dict["epoch_picks"] = os.path.join(base_path, mne_fdr, f"{subject}", f"{subject}" + "_picks_{epoch_key}.pkl")
    
    def get_pkl(self, epoch_key, dict_key):
        pkl_pth = self.path_dict[dict_key].format(epoch_key=epoch_key)
        with open(pkl_pth, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
    def print_pcs(self, epoch_key, target_dir, picks='Cz', window_length=1):
        pc = self.get_pkl(epoch_key, 'epoch_pc')
        
        ensure_dir(target_dir)
        ch_list = pick_indices(self.raw, picks, return_indices=False).ch_names
        pcs_plot(pc, target_fdr=target_dir, ch_list=ch_list, ch_names=pick_indices(self.raw, self.get_pkl(epoch_key, 'epoch_picks'), return_indices=False).ch_names, win_list=np.arange(window_length-1, pc.shape[0]), info=self.raw.info)
        
    # def print_noise_components(self, epoch_key, target_dir, picks='all', window_length=30):
    #     pcs = self.get_pcs(epoch_key)
    #     epoch = self.get_ep(epoch_key)
    #     orig_data = torch.tensor(epoch.get_data(picks=picks))        
        
    #     noise_components = []
        
    #     if len(pcs.shape) == 3:             # #ch, len(ep), #pc
    #         spurious_data
    #         coord = lstsq(pcs, noise.get_data())[0]    # #ch, #pc, #ep
    #         for i in range(pcs.shape[-1]):
    #             noise_components.append(pcs[:, :, i:i+1] @ coord[:, i:i+1])
    #     elif len(pcs.shape) == 4:           # 29+#win, #ch, len(ep), #pc
    #         spurious_data = orig_data - torch.mean(orig_data, dim=2).unsqueeze(2).unfold(0, window_length, 1)
    #         coord = lstsq(pcs, spurious_data)[0].unsqueeze(-1) # 29+#win, #ch, #pc, 1
    #         for i in range(pcs.shape[-1]):
    #             noise_components.append(pcs[:, :, :, i:i+1] @ coord[:, :, i:i+1])
                
    #     noise_dataset = copy.deepcopy(self.raw)
    #     for idx, noise_component in enumerate(noise_components):
    #         noise_dataset = mne_epoch2raw(epoch, noise_dataset, noise_component, overwrite='even', picks=picks)
    #         psd_plot([noise_dataset], [f"No. {idx} PC"], fs=self.raw.info['sfreq'], save_pth=os.path.join(target_dir, f"psd_{idx}.png"), picks=picks)
    #         temp_plot(noise_dataset, 'Cz', fs=self.raw.info['sfreq'], save_pth=os.path.join(target_dir, f"temp_Cz.png"), name='Cz')
    #         temp_plot(noise_dataset, 'Cz', start=100*self.raw.info['sfreq'], length=10*self.raw.info['sfreq'], fs=self.raw.info['sfreq'], save_pth=os.path.join(target_dir, f"temp_Cz.png"), name='Cz')
            
            
    #     return noise_components
    
    
