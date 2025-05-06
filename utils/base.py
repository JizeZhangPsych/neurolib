import os, re
from abc import ABC, abstractmethod

class BasePathfinder(ABC):
    def __init__(self, src, raw, pattern, tgt=None, **kwargs):
    
        if tgt is None:
            tgt = src
    
        self.fdr = {}
        self.fdr["tgt"] = tgt
        self.fdr["src"] = src

        self.fdr["raw"] = os.path.join(self.fdr['src'], raw)
        
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
