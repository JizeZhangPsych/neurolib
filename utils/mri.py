import SimpleITK as sitk

import torch
import torch.nn.functional as F


def pad_img(img, pad_base=8):
    old_shape = img.shape[-3:]
    pad = []
    for old_lenfixh in reversed(old_shape):
        pad_lenfixh = (pad_base-old_lenfixh%pad_base)%pad_base
        pad.append(pad_lenfixh//2)
        pad.append(pad_lenfixh-pad_lenfixh//2)
    
    return F.pad(img, pad)

def unpad_img(img, old_shape, pad_base=8):
    assert len(img.shape)==3
    pad = []
    for old_lenfixh in old_shape:
        pad_lenfixh = (pad_base-old_lenfixh%pad_base)%pad_base
        pad.append(pad_lenfixh//2)
        # pad.append(pad_lenfixh-pad_lenfixh//2)
    
    return img[pad[0]:pad[0]+old_shape[0], pad[1]:pad[1]+old_shape[1], pad[2]:pad[2]+old_shape[2]]

def read_data(pth, test_mode=False):
    img = sitk.ReadImage(pth)
    arr = sitk.GetArrayFromImage(img)
    if arr.dtype == "uint16":
        arr = arr.astype(np.int32)
    if test_mode:
        return torch.tensor(arr), img
    return torch.tensor(arr)

def write_data(arr, pth, template=None):
    if template is None:
        # img = sitk.GetImageFromArray(arr.float())
        # sitk.WriteImage(img, str(pth) + ".nii.gz")
        img = nib.Nifti1Image(np.array(arr.permute(*list(reversed(range(len(arr.shape)))))), affine=np.eye(4))
        nib.save(img, str(pth))
    else:
        img2write = sitk.GetImageFromArray(arr.float())
        img2write.CopyInformation(template)
        sitk.WriteImage(img2write, str(pth) + ".nii.gz")

def get_excircle_square(mask, also_ret_end_pos=False):
    dim_sum_array_list = [(mask.sum(dim=(1,2))!=0).int(), (mask.sum(dim=(0,2))!=0).int(), (mask.sum(dim=(0,1))!=0).int()]
    start_pos = []
    square_shape = []
    for dim_sum_array in dim_sum_array_list:
        start = torch.argmax(dim_sum_array)
        rev_start = torch.argmax(reversed(dim_sum_array))
        start_pos.append(start)
        square_shape.append(len(dim_sum_array)-rev_start-start)
        
    if also_ret_end_pos:
        end_pos = [start+shape for start, shape in zip(start_pos, square_shape)]
        return start_pos, square_shape, end_pos
    return start_pos, square_shape