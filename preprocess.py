import os
from skimage.transform import resize
import nibabel as nib
import numpy as np
import sys
import glob
from tqdm import tqdm
import pdb

########################################################################################################################
# Code for preprocessing all of the scans and storing them in numpy arrays. Done so that preprocessing wouldn't be
# repeated during training, thereby speeding up the training process. Assumes the data is all in the specified folder
# and not within sub folders in it (ie <Brats_ID> folder that stores the scans for that Brats_ID is in that folder and
# not a subfolder). Preprocessing first crops each image to keep the brain region only if specified after which the pixel
# values are standardized and scaled to the range 0 and 1.

# INPUT arguments:
#   arg1: path to where the raw scans to preprocess are stored
#   arg2: path where to save the preprocessed scans
#   arg3: Specify 1 if to crop images to exclude zero regions outside brain. 0 if pixel intensity scaling only.
#   arg4: Specify 1 if to preprocess training data (segmentation labels as well) or 0 if test data only.
#
# OUTPUT:
#   Preprocessed data stored in compressed npz files in the save_preprocessed_path
########################################################################################################################


crop_data = False

dataset = 'Brats18val'

if dataset == 'Brats19train':
    raw_data_path = '/data/xwj_work/Brats2019/raw/MICCAI_BraTS_2019_Data_Training/'
    save_preprocessed_data_path = '/data/xwj_work/Brats2019/train/'
    subdirs = glob.glob(raw_data_path + '/HGG/*') + glob.glob(raw_data_path + '/LGG/*')
    train_data = True
elif dataset == 'Brats19val':
    raw_data_path = '/data/xwj_work/Brats2019/raw/MICCAI_BraTS_2019_Data_Validation/'
    save_preprocessed_data_path = '/data/xwj_work/Brats2019/val/'
    subdirs = glob.glob(raw_data_path + '/Brats*') + glob.glob(raw_data_path + '/Brats*')
    train_data = False

elif dataset == 'Brats18train':
    raw_data_path = '/data/xwj_work/Brats2018/'
    save_preprocessed_data_path = '/data/xwj_work/Brats2018/train_nocrop/'
    subdirs = glob.glob(raw_data_path + 'HGG/*') + glob.glob(raw_data_path + 'LGG/*')
    train_data = True
elif dataset == 'Brats18val':
    raw_data_path = '/data/xwj_work/Brats2018/MICCAI_BraTS_2018_Data_Validation/'
    save_preprocessed_data_path = '/data/xwj_work/Brats2018/val_nocrop/'
    subdirs = glob.glob(raw_data_path + '/Brats*')
    train_data = False

elif dataset == 'Brats20train':
    raw_data_path = '/data/xwj_work/Brats2020/raw'
    save_preprocessed_data_path = '/data/xwj_work/Brats2020/train/'
    subdirs = glob.glob(raw_data_path + '/Brats*')
    train_data = True
pdb.set_trace()
# elif dataset == 'Brats20val':
#     raw_data_path = '/data/xwj_work/Brats2020/'
#     save_preprocessed_data_path = '/data/xwj_work/Brats2020/train/'
#     subdirs = glob.glob(raw_data_path + '/Brats*')
#     train_data = False

# Create the folder to store preprocessed data in, exit if folder already exists.
os.makedirs(save_preprocessed_data_path,exist_ok=True)

for subdir in tqdm(subdirs):
    # print (subdir)
    # Load in the the different modalities
    print(subdir)
    img_t1 = nib.load(glob.glob(subdir + "/*_t1.nii.gz")[0]).get_fdata()
    img_t1ce = nib.load(glob.glob(subdir + "/*_t1ce.nii.gz")[0]).get_fdata()
    img_t2 = nib.load(glob.glob(subdir + "/*_t2.nii.gz")[0]).get_fdata()
    img_flair = nib.load(glob.glob(subdir + "/*_flair.nii.gz")[0]).get_fdata()

    # If preprocessing training data, load in the segmentation label image too
    if train_data:
        img_seg = nib.load(glob.glob(subdir + "/*_seg.nii.gz")[0]).get_fdata().astype('long')
        img_seg[img_seg == 4] = 3  # Replace label 4 with label 3

    h, w, d = img_t1.shape
    r = np.any(img_t1, axis=(1, 2))
    c = np.any(img_t1, axis=(0, 2))
    z = np.any(img_t1, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    if (rmax - rmin < 127):
        diff = 127 - (rmax - rmin)
        pad_left = int(diff / 2)
        pad_right = diff - pad_left
        rmin = rmin - pad_left
        rmax = rmax + pad_right
        if rmax == d:
            rmin = rmin - 1
        if rmin == 0:
            rmax = rmax + 1

    if (cmax - cmin < 127):
        diff = 127 - (cmax - cmin)
        pad_left = int(diff / 2)
        pad_right = diff - pad_left
        cmin = cmin - pad_left
        cmax = cmax + pad_right
        if cmax == d:
            cmin = cmin - 1
        if cmin == 0:
            cmax = cmax + 1

    if (zmax - zmin < 127):
        diff = 127 - (zmax - zmin)
        pad_left = int(diff / 2)
        pad_right = diff - pad_left
        zmin = zmin - pad_left
        zmax = zmax + pad_right
        if zmax == d:
            zmin = zmin - 1
        if zmin == 0:
            zmax = zmax + 1

    nozero = {'h': [rmin, rmax + 1], 'w': [cmin, cmax + 1], 'd': [zmin, zmax + 1]}
    # Crop the images to only keep bounding box area of the brain, with the bounding box atleast 128 length in each dimension
    if crop_data:
        img_t1 = img_t1[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
        img_t1ce = img_t1ce[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
        img_t2 = img_t2[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
        img_flair = img_flair[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
        img_seg = img_seg[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]

    # Standardize and scale pixel intensity values and store all the modalities in the same array
    X = []
    for modality in [img_t1, img_t1ce, img_t2, img_flair]:
        brain_region = modality > 0  # Get region of brain to only manipulate those voxels
        mean = np.mean(modality[brain_region])
        stdev = np.std(modality[brain_region])
        new_img = np.zeros(img_t1.shape)
        new_img[brain_region] = (modality[brain_region] - mean) / stdev  # Standardize by mean and stdev
        # new_img[new_img > 5] = 5  # Clip outliers
        # new_img[new_img < -5] = -5
        Maximum = np.max(new_img)
        Minimum = np.min(new_img[brain_region])
        Range = Maximum - Minimum
        new_img[brain_region] = (new_img[brain_region] - Minimum) / Range  # Scale to be between 0 and 1
        X.append(new_img.astype('float32'))

    save_scans = save_preprocessed_data_path + subdir.split('/')[-1] + '_scans.npz'
    np.savez_compressed(save_scans, data=X, nozero=nozero)
    if train_data:
        save_mask = save_preprocessed_data_path + subdir.split('/')[-1] + '_mask.npz'
        np.savez_compressed(save_mask, data=img_seg, nozero=nozero)
