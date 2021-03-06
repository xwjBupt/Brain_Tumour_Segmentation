import SimpleITK as sitk
import nibabel as nib
import os
import pandas as pd
import sys
from sys import argv
import numpy as np

########################################################################################################################
# Code for evaluating the segmentation performance of a model (ensemble averaged output over folds). Returns Dice scores
# for the whole tumor, the enhancing and the necrotic core regions for all the folds as well as the average over the folds.
# Saves the results in an excel sheet. When calling script:

# INPUT Arguments:
#   arg1: path to where the segmented scans are stored
#   arg2: path to where the original brats folders for those scans are stored
#   arg3: path where to store excel sheet of results for current run
#   arg4: nothing or 1: whether to add mean dice score to All_Results.xlsx file if one wants to store the results of all the
#         runs in a same file.

# OUTPUT:
#   Excel sheet of segmentation scores stored in the provided save results path
########################################################################################################################

segmented_data_path = argv[1]  # '/home/artur-cmic/Desktop/UCL/Data_Aug_Experiments/Data/temp/Affine_010_1'
original_data_path = argv[2]  # '/home/artur-cmic/Desktop/UCL/Data_Aug_Experiments/Data/James_Validation'
save_results_path = argv[3]  # '/home/artur-cmic/Desktop/UCL/Data_Aug_Experiments/Results_James'
all_runs_path = 1
if len(argv) > 4:
    all_runs_path = argv[4]
run_name = segmented_data_path[segmented_data_path.rindex("/") + 1:]
print("Getting Results for run " + run_name)

# Get the paths of the files in the segmented scans folder
seg_paths = []
seg_ids = []
for file_to_segment in os.listdir(segmented_data_path):
    seg_paths.append(os.path.join(segmented_data_path, file_to_segment))
    seg_ids.append(file_to_segment[:-7])

results = {}
for image in range(len(seg_paths)):
    ground_truth = nib.load(
        os.path.join(original_data_path, seg_ids[image], seg_ids[image]) + "_seg.nii.gz").get_data().astype('uint8')
    prediction = nib.load(seg_paths[image]).get_data().astype('uint8')

    whole_tumor_gt = (ground_truth > 0).astype('int')
    enhancing_gt = (ground_truth == 4).astype('int')
    core_gt = (ground_truth == 1).astype('int')

    whole_tumor_pred = (prediction > 0).astype('int')
    enhancing_pred = (prediction == 4).astype('int')
    core_pred = (prediction == 1).astype('int')

    comparisons = {'Whole': {'pred': whole_tumor_pred, 'gt': whole_tumor_gt},
                   'Enhancing': {'pred': enhancing_pred, 'gt': enhancing_gt},
                   'Core': {'pred': core_pred, 'gt': core_gt}}
    measures = {}
    for region_name, masks in comparisons.items():
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(sitk.GetImageFromArray(masks['pred']), sitk.GetImageFromArray(masks['gt']))
        overlap = overlap_measures_filter.GetDiceCoefficient()
        # If a subregion of the tumor was not present in the labels / image, then overlap will be equal to infinity as there is nothing to compare
        # and mean over all images would become infinity. For these cases make overlap equal to NaN so it is not included in the mean calculations.
        if (overlap > 1):
            overlap = np.nan
        measures[region_name] = overlap
    results[seg_ids[image]] = measures

results = pd.DataFrame.from_dict(results, orient='index').astype('float')
res_mean = results.mean(axis=0, skipna=True)
res_std = results.std(axis=0, skipna=True)
results.loc['Mean'] = res_mean
results.loc['Std'] = res_std
results = results.astype('float').round(3)
writer = pd.ExcelWriter('{}/overlap_{}.xlsx'.format(save_results_path, run_name), engine='xlsxwriter')
results.to_excel(writer)
writer.save()

if all_runs_path:
    if not os.path.isfile(os.path.join(save_results_path, "All_Results.xlsx")):
        column_index = [np.array(['Run', 'Mean', 'Mean', 'Mean', 'Std', 'Std', 'Std']),
                        np.array([''] + res_mean.index.tolist() + res_mean.index.tolist())]
        values = [run_name] + res_mean.round(3).values.tolist() + res_std.round(3).values.tolist()
        all_results = pd.DataFrame(values, index=column_index).transpose()
        all_results.set_index('Run', inplace=True, drop=True)
        writer = pd.ExcelWriter('{}/All_Results.xlsx'.format(save_results_path), engine='xlsxwriter')
        all_results.to_excel(writer)
        writer.save()
    else:
        all_results = pd.read_excel('{}/All_Results.xlsx'.format(save_results_path), header=[0, 1], index_col=0)
        column_index = [np.array(['Mean', 'Mean', 'Mean', 'Std', 'Std', 'Std']),
                        np.array(res_mean.index.tolist() + res_mean.index.tolist())]
        values = res_mean.round(3).values.tolist() + res_std.round(3).values.tolist()
        all = pd.Series(values, index=column_index)
        all_results.loc[run_name] = all
        writer = pd.ExcelWriter('{}/All_Results.xlsx'.format(save_results_path), engine='xlsxwriter')
        all_results.to_excel(writer)
        writer.save()
