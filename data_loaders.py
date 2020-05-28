import numpy as np
from torch.utils import data
import random
import glob
import os
import pdb

##########################################################################
# Dataset class that feeds data into a data generator.
# INPUT:
#   folder_path: list of full folder paths of the preprocessed data.
#   folder_id: list of names of those folders.
#   seg_provided: specify False if performing inference and only scans should be loaded.
#
# OUTPUT:
#   X, y numpy arrays, where X = [C,H,W,D] contains C modality scans and y = [H,W,D] contains the segmentation labels,
#   both randomly sampled 128x128x128 patches from the input image.
##########################################################################


class Dataset(data.Dataset):
    def __init__(self, folder_ids, seg_provided=True, nozero=True):
        self.folder_ids = folder_ids
        self.seg_provided = seg_provided
        self.nozero = nozero

    def __len__(self):
        return len(self.folder_ids)

    def __getitem__(self, index):
        data_id = self.folder_ids[index]
        X = np.load(data_id)['data']
        nozero_range = np.load(data_id)['nozero'].item()
        x_orig, y_orig, z_orig = 0, 0, 0
        x_stop, y_stop, z_stop = X.shape[1], X.shape[2], X.shape[3]

        if self.nozero:
            hmin, hmax = nozero_range['h'][0],nozero_range['h'][1]
            wmin, wmax = nozero_range['w'][0],nozero_range['w'][1]
            dmin, dmax = nozero_range['d'][0],nozero_range['d'][1]
            x_orig, y_orig, z_orig = hmin, wmin, dmin
            x_stop, y_stop, z_stop = hmax, wmax, dmax

            # Randomly sample 128x128x128 patch
            if ((x_stop - x_orig) > 128):
                x_orig = random.sample(range(x_stop - 127), 1)[0]
            if ((y_stop - y_orig) > 128):
                y_orig = random.sample(range(y_stop - 127), 1)[0]
            if ((z_stop - z_orig) > 128):
                z_orig = random.sample(range(z_stop - 127), 1)[0]

            X = X[:, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]

        if self.seg_provided:
            y = np.load(data_id.replace('scans.npz', 'mask.npz'))['data']
            y = y[x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]
            return X, y
        else:
            return X


if __name__ == '__main__':
    folder_paths = []
    folder_ids = []
    preprocessed_data_path = '/Users/xwj/Downloads/BT/BraTs2018/processed'
    for subdir in os.listdir(preprocessed_data_path):
        folder_paths.append(os.path.join(preprocessed_data_path, subdir))
        folder_ids.append(subdir)
    D = Dataset(folder_paths, folder_id=folder_ids)
    for i in enumerate(D):
        print('Done')
