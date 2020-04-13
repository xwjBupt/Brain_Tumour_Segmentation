import numpy as np
from torch.utils import data
import random
import glob
import os


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
    def __init__(self, folder_path, folder_id, seg_provided=True):
        self.folder_paths = folder_path
        self.folder_ids = folder_id
        self.seg_provided = seg_provided

    def __len__(self):
        return len(self.folder_ids)

    def __getitem__(self, index):
        data_folder = self.folder_paths[index]
        data_id = self.folder_ids[index]
        X = np.load(glob.glob(r"{}/{}*_scans.npz".format(data_folder, data_id))[0])['data']
        x_orig, y_orig, z_orig = 0, 0, 0

        # Randomly sample 128x128x128 patch
        if (X.shape[1] > 128):
            x_orig = random.sample(range(X.shape[1] - 127), 1)[0]
        if (X.shape[2] > 128):
            y_orig = random.sample(range(X.shape[2] - 127), 1)[0]
        if (X.shape[3] > 128):
            z_orig = random.sample(range(X.shape[3] - 127), 1)[0]

        X = X[:, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]

        if self.seg_provided:
            y = np.load(glob.glob(r"{}/{}*_mask.npz".format(data_folder, data_id))[0])['data']
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
