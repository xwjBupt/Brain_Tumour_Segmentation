import torch
import numpy as np
import random
import os
from torch.utils import data
from sklearn.model_selection import KFold
from data_loaders import Dataset
from data_augment import DataAugment
import torch.nn as nn
import time
from torch.optim import lr_scheduler

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import io
import PIL.Image
import sys
import copy
from torchvision.transforms import ToTensor
import matplotlib
import pdb
import glob

matplotlib.use('agg')
from matplotlib import pyplot as plt
from collections import OrderedDict
from model_utils.utils import expand_as_one_hot
import time
# Specify Which Model and Loss to Import
from models import UNet3D
from losses import GeneralizedDiceLoss
from HDCUnet import *

#
# def gen_subplot(scans_orig, scans_aug, mask, prediction, epoch_nr, name):
#     """
#     :param scans_orig: shape[B,C,H,W,D] torch.tensor
#     :param scans_aug:shape[B,C,H,W,D] torch.cuda.tensor
#     :param mask:
#     :param prediction:
#     :param epoch_nr:
#     :param aug_used:
#     :return:
#     """
#
#     def gettogthoer(mask):
#         # mask[C,H,W,D]
#         # whole [H,W,D]
#         wt = mask[0]
#         tc = mask[1]
#         et = mask[2]
#         whole = np.zeros(mask[0].shape)
#         whole[wt > 0.5] = 2
#         whole[tc > 0.5] = 1
#         whole[et > 0.5] = 4
#         whole = whole.astype("uint8")
#         return whole
#
#     slices = scans_aug.shape[-1]
#     temp = slices // 5
#     scans_orig = scans_orig[0][0].cpu().detach().numpy()
#     scans_aug = scans_aug[0][0].cpu().detach().numpy()
#     mask = gettogthoer(mask[0].cpu().detach().numpy())
#     prediction = gettogthoer(prediction[0].cpu().detach().numpy())
#     slice_num = [0 * temp, temp * 2 - 10, temp * 2, temp * 2 + 10, temp * 3]
#     plt.figure()
#
#     for i in range(4):
#         if i == 0:
#             show = scans_orig
#         if i == 1:
#             show = scans_aug
#         if i == 2:
#             show = mask
#         if i == 3:
#             show = prediction
#
#         plt.subplot(4, 5, 1 + i * 5)
#         plt.title('slice:' + str(slice_num[0])) if i == 0 else None
#
#         if i == 0:
#             plt.ylabel('scans_orig')
#         elif i == 1:
#             plt.ylabel('scans_aug')
#         elif i == 2:
#             plt.ylabel('mask')
#         elif i == 3:
#             plt.ylabel('prediction')
#
#         plt.imshow(show[:, :, slice_num[0]])
#
#         plt.subplot(4, 5, 2 + i * 5)
#         plt.title('slice:' + str(slice_num[1])) if i == 0 else None
#         plt.imshow(show[:, :, slice_num[1]])
#
#         plt.subplot(4, 5, 3 + i * 5)
#         plt.title('slice:' + str(slice_num[2])) if i == 0 else None
#         plt.imshow(show[:, :, slice_num[2]])
#
#         plt.subplot(4, 5, 4 + i * 5)
#         plt.title('slice:' + str(slice_num[3])) if i == 0 else None
#         plt.imshow(show[:, :, slice_num[3]])
#
#         plt.subplot(4, 5, 5 + i * 5)
#         plt.title('slice:' + str(slice_num[4])) if i == 0 else None
#         plt.imshow(show[:, :, slice_num[4]])
#
#         plt.suptitle("%s @ epoch:%d " % (name.split('/')[-1], epoch_nr))
#
#     plt.savefig(name + '@' + str(epoch_nr) + '.png')
#     plt.close()


def gen_subplot(scans_orig, scans_aug, mask, prediction, epoch_nr, iteration, aug_used):
    _, indices = prediction.max(0)
    indices = indices.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()
    img_size = indices.shape[2]
    slices = [int(img_size / 4), int(img_size / 4 * 2), int(img_size / 4 * 3)]  # Slices to display
    plt.figure()
    for row in range(1, 4):
        plt.subplot(3, 8, 1 + (row - 1) * 8)
        # Showing the t1ce scan
        plt.imshow(scans_orig[1, :, :, slices[row - 1]], cmap='gray')
        plt.ylabel("Slice {}".format(slices[row - 1]))
        if row == 1:
            plt.title('Orig')
        plt.subplot(3, 8, 2 + (row - 1) * 8)
        # Showing the t1ce scan
        plt.imshow(scans_aug[1, :, :, slices[row - 1]], cmap='gray')
        plt.ylabel("Slice {}".format(slices[row - 1]))
        if row == 1:
            plt.title('Aug: ' + aug_used)
        plt.subplot(3, 8, 3 + (row - 1) * 8)
        plt.imshow(mask[:, :, slices[row - 1]])
        if row == 1:
            plt.title('Mask')
        plt.subplot(3, 8, 4 + (row - 1) * 8)
        plt.imshow(prediction[0, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 0')

        plt.subplot(3, 8, 5 + (row - 1) * 8)
        plt.imshow(prediction[1, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 1')

        plt.subplot(3, 8, 6 + (row - 1) * 8)
        plt.imshow(prediction[2, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 2')

        plt.subplot(3, 8, 7 + (row - 1) * 8)
        plt.imshow(prediction[3, :, :, slices[row - 1]])
        if row == 1:
            plt.title('Cl 3')

        plt.subplot(3, 8, 8 + (row - 1) * 8)
        plt.imshow(indices[:, :, slices[row - 1]])
        if row == 1:
            plt.title('Pred')
    plt.suptitle("Epoch {} Iteration {}".format(str(epoch_nr), str(iteration)))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image


########################################################################################################################
# Code for training a 3D-Unet, models and losses must be specified. Currently uses 3D-UNet and Generalized dice loss
# obtained form wolny/pytorch-3dunet

# To specify:

# Paths where to load data from and save the models to
preprocessed_data_path = r'/data/xwj_work/Brats2018/train_nocrop'
save_model_path = r"/data/xwj_work/Brain_Tumour_Segmentation/train_nocrop"

# Specify which data augmentations to use on the fly (each applied with 50% probability). Possible values:
# ['Elastic', 'Flip', 'Rotate','Gamma','Scale', 'Noise']. Create empty array if none wanted.
augmentations_to_use = ['Flip']  # 'Flip', 'Rotate', 'Gamma', 'Scale', 'Noise']
timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
# Name of the run
run_name = "temp_" + timestamp
datasetname = 'Brats2018_val2019'
# Training Parameters
batch_size = 1
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8}
max_epochs = 300

# Model Parameters
in_channels = 4
n_classes = 4
base_n_filter = 16
n_folds = 2  # Number of folds in cross-validation

##############################################################################

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/' + run_name)

if not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)

# Use GPU
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Get paths and names (IDS) of folders that store the multimodal training data
folder_paths = glob.glob(preprocessed_data_path + '/*scans.npz')
# folder_paths = []
# folder_ids = []
# for subdir in os.listdir(preprocessed_data_path):
#     folder_paths.append(os.path.join(preprocessed_data_path, subdir))
#     folder_ids.append(subdir)

# Shuffle them around, keeping same seed to make sure same shuffling is used if training is interrupted and needs to be continued
random.seed(675)
random.shuffle(folder_paths)
# random.seed(4)
# random.shuffle(folder_ids)

# Setup KFold Cross Validation
# kf = KFold(n_splits=n_folds, shuffle=False)  # Shuffle=false to get the same shuffling scheme every run
# fold_nr = 1

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if datasetname != 'Brats2018_val2019':
    samples = len(folder_paths)
    val_num = int(samples * 0.2)
    train_num = samples - val_num
    print('creating %d for train %d for val' % (train_num, val_num))
    split = {}
    for i in range(n_folds):
        f = {}
        flodname = 'fold' + str(i + 1)
        f['train'] = folder_paths[val_num:]
        f['val'] = folder_paths[:val_num]
        split[flodname] = f
    savename = 'Brats2018' + '@' + timestamp + '.npy'
    np.save(savename, split)
else:
    split = {}
    temptrainlist = glob.glob('/data/xwj_work/Brats2018/' + 'train/' + '*scans.npz')
    tempvallist = glob.glob('/data/xwj_work/Brats2018/' + 'val2019/' + '*scans.npz')
    split['fold_all'] = {'train': temptrainlist, 'val': tempvallist}

for fold, files in split.items():
    iter_nr = 1
    train_idx = files['train']
    valid_idx = files['val']
    train_set = Dataset(train_idx)
    valid_set = Dataset(valid_idx)
    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # Model
    #model = UNet3D(in_channels, n_classes, False, base_n_filter, 'crg', 8)

    model = HDCUnetV2(in_channels=4, out_channels=4, init_channels=16, bias=False, skip=True, norm='G', act='relu',
                    sig=True, gtcount=False, inside_skip='CAT')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # If training was interrupted (need to change epoch loop range as well):
    # checkpoint = torch.load("/home/ajurgens/Brats2019/Model_Saves_V4/Fold_1_Epoch_140.tar")
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Loss and optimizer
    criterion = GeneralizedDiceLoss(1e-5, None, None, False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=10 ** -5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
                                               factor=0.5,
                                               patience=20,
                                               min_lr=5e-6)
    model.cuda()

    # If training was interrupted (need to change epoch loop range as well):
    model.train()

    for epoch in range(max_epochs + 1):
        start_time = time.time()
        train_losses = []
        print('Epoch %d Training' % (epoch))
        for batch, labels in tqdm(train_loader):
            batch_orig = copy.deepcopy(batch)  # Save batch before augmentation for plotting outputs

            # Data Augment if augmentations were given
            if not len(augmentations_to_use) == 0:
                augmenter = DataAugment(batch, labels, augmentations_to_use, True)
                batch, labels, augmentation_parameters = augmenter.augment()

            # If scaling to a bigger volume was performed, randomly sample a 128x128x128 patch again or it wont fit into GPU memory
            if batch.shape[2] > 128:
                x_orig = random.sample(range(batch.shape[2] - 127), 1)[0]
                y_orig = random.sample(range(batch.shape[3] - 127), 1)[0]
                z_orig = random.sample(range(batch.shape[4] - 127), 1)[0]
                batch = batch[:, :, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]
                labels = labels[:, x_orig: x_orig + 128, y_orig: y_orig + 128, z_orig: z_orig + 128]

            # Transfer batch and labels to GPU
            scans, masks = batch.cuda(), labels.cuda()
            # Run through network
            output = model(scans)
            pdb.set_trace()
            masks = expand_as_one_hot(masks, n_classes)
            train_loss = criterion(output, masks)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            # Save images of network output every 100 iterations
            if (iter_nr % 100 == 0):
                subplot_img = gen_subplot(batch_orig[0], batch[0], labels[0], output[0], epoch, iter_nr,
                                          augmentation_parameters)
                writer.add_image('{}'.format(run_name), subplot_img, iter_nr)

            # Log training loss to tensorboard
            writer.add_scalar('Train Loss', train_loss.item(), iter_nr)
            iter_nr += 1

        # Get training loss after every epoch
        train_loss_ep = np.mean(train_losses)
        writer.add_scalar('TrainPE Fold {}'.format(fold), train_loss_ep, epoch)
        valid_losses = []
        with torch.no_grad():
            model.eval()
            print('Epoch %d Valing' % (epoch))
            for batch, labels in tqdm(valid_loader):
                scans, masks = batch.cuda(), labels.cuda()
                output = model(scans)
                masks = expand_as_one_hot(masks, n_classes)
                valid_loss = criterion(output, masks)
                valid_losses.append(valid_loss.item())
        valid_loss_ep = np.mean(valid_losses)
        # Log valid loss to tensorboard
        writer.add_scalar('Valid Loss per Epoch', valid_loss_ep, epoch)
        elapsed_time = time.time() - start_time
        scheduler.step(valid_loss_ep)
        print('Fold [{}], Epoch [{}/{}], Train Loss {:.10f}, Valid Loss {:.10f}, Time_{}\n'.format(fold,
                                                                                                   epoch, max_epochs,
                                                                                                   train_loss_ep,
                                                                                                   valid_loss_ep,
                                                                                                   time.strftime(
                                                                                                       "%H:%M:%S",
                                                                                                       time.gmtime(
                                                                                                           elapsed_time))))
        losses = open("{}/Losses_{}.txt".format(save_model_path, run_name), "a")
        losses.write(
            'Fold [{}], Epoch [{}/{}], Train Loss {:.10f}, Valid Loss {:.10f}, Time {}\n'.format(fold,
                                                                                                 epoch, max_epochs,
                                                                                                 train_loss_ep,
                                                                                                 valid_loss_ep,
                                                                                                 time.strftime(
                                                                                                     "%H:%M:%S",
                                                                                                     time.gmtime(
                                                                                                         elapsed_time))))
        losses.close()

        # Save the model parameters
        if (epoch % 10 == 0):
            torch.save(model, "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold, epoch))
            # if torch.cuda.device_count() > 1:
            #     torch.save(
            #         {'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            #         "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold, epoch))
            # else:
            #     torch.save(
            #         {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
            #         "{}/Fold_{}_Epoch_{}.tar".format(save_model_path, fold, epoch))
