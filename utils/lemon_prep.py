import os, re

def lemon_init(dataset, userargs):
    dataset['pf'] = LemonPathFinder(**userargs)
    return dataset


class LemonPathFinder:
    def __init__(self, tgt="/well/woolrich/users/kcq472/lemon", src="/well/woolrich/projects/lemon", prep="after_prep", recon="after_recon", hmm="after_hmm", raw="raw", debug="debug", pattern=r"^sub-\d{6}$", **kwargs):
        if tgt is None:
            tgt = src
    
        self.fdr = {}
        self.fdr["tgt"] = tgt
        self.fdr["src"] = src

        self.fdr["raw"] = os.path.join(self.fdr['src'], raw)
        
        self.fdr["prep"] = os.path.join(self.fdr['tgt'], prep)
        self.fdr["recon"] = os.path.join(self.fdr['tgt'], recon)
        self.fdr["hmm"] = os.path.join(self.fdr['tgt'], hmm)
        self.fdr["debug"] = os.path.join(self.fdr['tgt'], debug)
        
        for key, value in kwargs.items():
            if key in self.fdr.keys():
                raise ValueError(f"Key {key} already exists in fdr dictionary, don't include it in kwargs. Existing keys are {self.fdr.keys()}")
            self.fdr[key] = os.path.join(self.fdr['base'], value)
        
        self.subject_list = []
        for filename in os.listdir(self.fdr["raw"]):
            match = re.match(pattern, filename)
            if match:
                groups = match.groups()
                if not groups:
                    self.subject_list.append(match.group(0))
                else:
                    self.subject_list.append("_".join(groups))
                    
    def get_fdr_dict(self):
        return self.fdr
    
    def get_eeg_file(self, subj_str, data_type, postfix):
        """
            Returns the path to the file given the subject string and the data type.
            Parameters:
            subj_str (str): Subject string.
            data_type (str): Data type. Must be one of the keys in Pathfinder.fdr.
        """
        assert data_type in self.fdr.keys(), f"Data type {data_type} not found in fdr dictionary. Available keys are {self.fdr.keys()}"
        subject = "sub-" + subj_str
        if subj_str == 'raw':
            raw_file = os.path.join(self.fdr["raw"], f"{subject}/RSEEG/{subject}.vhdr")
            return raw_file
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented. Available keys are {self.fdr.keys()}")