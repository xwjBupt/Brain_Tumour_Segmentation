import numpy as np
import nibabel as nib
from data_loaders import Dataset
from models import UNet3D
from skimage.transform import resize
import os
import torch
from collections import OrderedDict
from sys import argv
import glob
import pdb

########################################################################################################################
# Code to segment a set of scans after model has been trained. Segmentation obtained by ensemble averaging outputs of the
# models from the different folds. When calling script:

# INPUT arguments:
#   arg1: path to where the preprocessed scans to segment are stored
#   arg2: path to where the Model_Saves are stored
#   arg3: epoch nr at which to to load a models save
#   arg4: path where to save the segmented scans. Code creates folder with name of run in it

# OUTPUT:
#   segmented scans stored as nii.gz files in provided save results path
########################################################################################################################

preprocessed_data_path = '/data/xwj_work/Brats2018/val_nocrop/'
model_saves_path = '/data/xwj_work/Brain_Tumour_Segmentation/train_nocrop/'
epoch_nrs = [300, 290, 280]  # Epoch at which to take the model saves (determined from loss plots)
save_results_path = '/data/xwj_work/Brain_Tumour_Segmentation/val_nocrop_results/'

run_name = model_saves_path[model_saves_path.rindex("/") + 1:]
save_results_path = os.path.join(save_results_path, run_name)
if not os.path.isdir(save_results_path):
    os.mkdir(save_results_path)

# Get folder paths and names (IDS) of folders that store the preprocessed data
valist = glob.glob(preprocessed_data_path + '/Brats*')
valid_set = Dataset(valist, seg_provided=False, nozero=False)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
num_folds = 2
models = []
for epoch_nr in epoch_nrs:
    for fold in range(1, num_folds+1):
        model = UNet3D(4, 4, False, 16, 'crg', 8)
        print ("Loading {}/Fold_fold{}_Epoch_{}.tar".format(model_saves_path, fold, epoch_nr))
        checkpoint = torch.load("{}/Fold_fold{}_Epoch_{}.tar".format(model_saves_path, fold, epoch_nr))
#         if list(checkpoint['model_state_dict'].keys())[0].find("module") > -1:
#             new_state_dict = OrderedDict()
#             for k, v in checkpoint['model_state_dict'].items():
#                 name = k[7:]  # remove module.
#                 new_state_dict[name] = v
#             model.load_state_dict(new_state_dict)
#         else:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        models.append(model)
for idx in range(0, len(valist)):
    with torch.no_grad():
        scans = valid_set[idx]
        scans = np.expand_dims(scans, 0)
        scans = torch.from_numpy(scans).to(device)
        seg_results = []
        for fold in range(len(models)):
            output = models[fold](scans)
            output = output.squeeze()
            seg_results.append(output)
        seg_layer_avg = torch.mean(torch.stack(seg_results), dim=0)
        _, indices = seg_layer_avg.max(0)
        indices = indices.cpu().detach().numpy().astype('uint8')
        indices[indices == 3] = 4
        img = nib.Nifti1Image(indices, [[-1, -0, -0, 0], [-0, -1, -0, 239], [0, 0, 1, 0], [0, 0, 0, 1]])
        nib.save(img, "{}/{}.nii.gz".format(save_results_path, valist[idx].rsplit('/')[-1][:-10]))
        print('Saved example {}/{} for run {}'.format(idx + 1, len(valist), run_name))
