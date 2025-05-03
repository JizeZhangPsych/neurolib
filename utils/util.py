from pathlib import Path

def ensure_dir(dirname):
    """ make dir no matter if target folder & its parent folders exists
        Return True if target folder created.
        Return False if target folder already exists (nothing is done.)
    
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
        return True
    return False

def proc_userargs(userargs, default, strict=True):
    """
    Process user arguments and return a list of values based on the keys in userargs and default.
    Args:
        userargs (dict): User-defined arguments.
        default (dict): Default arguments.
        strict (bool): If True, raise KeyError for unexpected keys in userargs.
    Returns:
        list: A list of values corresponding to the keys in userargs and default.
    """
    return_dict = default.copy()
    for key in userargs.keys():
        if strict and key not in default:
            raise KeyError(f"Key {key} is not expected in userargs.")
        return_dict[key] = userargs[key]
    return return_dict