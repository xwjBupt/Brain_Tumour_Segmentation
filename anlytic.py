import matplotlib.pyplot as plt
import numpy as np
import cv2
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
import torch
import pdb
from tqdm import tqdm


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


def densiGt(mask, mode='all', sigma=0.75):
    temp = np.ones(mask.shape, dtype=np.float32)
    densi_avg = mask * scipy.ndimage.filters.gaussian_filter(temp, sigma, mode='constant')
    densi_mask_ss = np.zeros(mask.shape, dtype=np.float32)
    densi_mask_bs = np.zeros(mask.shape, dtype=np.float32)
    index = np.where(mask > 0.001)
    points = len(index[0])
    for i in range(points):
        temp = np.zeros(mask.shape, dtype=np.float32)
        temp[index[0][i], index[1][i]] = 1
        densi_mask_bs += scipy.ndimage.filters.gaussian_filter(temp, 1.25, mode='constant')
        densi_mask_ss += scipy.ndimage.filters.gaussian_filter(temp, 0.5, mode='constant')
    densi_avg = densi_avg[:, :, np.newaxis]
    densi_mask_bs = densi_mask_bs[:, :, np.newaxis]
    densi_mask_ss = densi_mask_ss[:, :, np.newaxis]
    return torch.tensor(densi_avg, dtype=torch.float32), torch.tensor(densi_mask_bs, dtype=torch.float32), torch.tensor(
        densi_mask_ss, dtype=torch.float32)


if __name__ == '__main__':
    root = '/home/xwj/Brats2018/processed/'
    maskfiles = glob.glob(root + '*/*mask.npz')
    for maskfile in tqdm(maskfiles):
        print(maskfile)
        WT_density_mask = {'avg': [], 'bs': [], 'ss': []}
        TC_density_mask = {'avg': [], 'bs': [], 'ss': []}
        ET_density_mask = {'avg': [], 'bs': [], 'ss': []}
        mask = np.load(maskfile)['data']
        h, w, d = mask.shape
        for i in range(d):
            WT_mask = (mask[:, :, i] > 0).astype('float')
            TC_mask = (mask[:, :, i] == 1).astype('float')
            ET_mask = (mask[:, :, i] == 3).astype('float')

            tempdensity = densiGt(WT_mask)
            WT_density_mask['avg'].append(tempdensity[0])
            WT_density_mask['bs'].append(tempdensity[1])
            WT_density_mask['ss'].append(tempdensity[2])

            tempdensity = densiGt(ET_mask)
            ET_density_mask['avg'].append(tempdensity[0])
            ET_density_mask['bs'].append(tempdensity[1])
            ET_density_mask['ss'].append(tempdensity[2])

            tempdensity = densiGt(TC_mask)
            TC_density_mask['avg'].append(tempdensity[0])
            TC_density_mask['bs'].append(tempdensity[1])
            TC_density_mask['ss'].append(tempdensity[2])

        WT_density_mask['avg'] = torch.cat(WT_density_mask['avg'], 2).numpy()
        WT_density_mask['bs'] = torch.cat(WT_density_mask['bs'], 2).numpy()
        WT_density_mask['ss'] = torch.cat(WT_density_mask['ss'], 2).numpy()

        ET_density_mask['avg'] = torch.cat(ET_density_mask['avg'], 2).numpy()
        ET_density_mask['bs'] = torch.cat(ET_density_mask['bs'], 2).numpy()
        ET_density_mask['ss'] = torch.cat(ET_density_mask['ss'], 2).numpy()

        TC_density_mask['avg'] = torch.cat(TC_density_mask['avg'], 2).numpy()
        TC_density_mask['bs'] = torch.cat(TC_density_mask['bs'], 2).numpy()
        TC_density_mask['ss'] = torch.cat(TC_density_mask['ss'], 2).numpy()

        np.savez_compressed(maskfile.replace('.npz', '_density.npz'), WT_density_mask=WT_density_mask,
                            TC_density_mask=TC_density_mask, ET_density_mask=ET_density_mask, rawmask=mask)

    # maskset = np.load('Brats18_CBICA_AUR_1_HGG_mask_density.npz')
    # WT_density_mask = maskset['WT_density_mask'].item()
    # TC_density_mask = maskset['TC_density_mask'].item()
    # ET_density_mask = maskset['ET_density_mask'].item()
    # a = [WT_density_mask, ET_density_mask, TC_density_mask]
    # h, w, d = ET_density_mask['bs'].shape
    # plt.ion()
    # for i in range(0, d, 5):
    #     plt.suptitle(str(i))
    #
    #     plt.subplot(331)
    #     plt.title('WT-AVG')
    #     plt.imshow(WT_density_mask['avg'][:, :, i])
    #
    #     plt.subplot(332)
    #     plt.title('WT-BS')
    #     plt.imshow(WT_density_mask['bs'][:, :, i])
    #
    #     plt.subplot(333)
    #     plt.title('WT_SS')
    #     plt.imshow(WT_density_mask['ss'][:, :, i])
    #
    #     plt.subplot(334)
    #     plt.title('ET-AVG')
    #     plt.imshow(ET_density_mask['avg'][:, :, i])
    #
    #     plt.subplot(335)
    #     plt.title('ET-BS')
    #     plt.imshow(ET_density_mask['bs'][:, :, i])
    #
    #     plt.subplot(336)
    #     plt.title('ET_SS')
    #     plt.imshow(ET_density_mask['ss'][:, :, i])
    #
    #     plt.subplot(337)
    #     plt.title('TC-AVG')
    #     plt.imshow(TC_density_mask['avg'][:, :, i])
    #
    #     plt.subplot(338)
    #     plt.title('TC-BS')
    #     plt.imshow(TC_density_mask['bs'][:, :, i])
    #
    #     plt.subplot(339)
    #     plt.title('TC_SS')
    #     plt.imshow(TC_density_mask['ss'][:, :, i])
    #     plt.pause(1)
    # print('DONE')
