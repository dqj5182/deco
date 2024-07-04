import os
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())

from utils.metrics import metric, precision_recall_f1score, det_error_metric


eval_outputs_path = 'test_outputs/DECO/damon/deco_damon_outputs_smplx.npz'
eval_outputs = np.load(eval_outputs_path)


eval_smpl_dict = {'cont_precision': [], 'cont_recall': [], 'cont_f1': [], 'fp_geo_err': [], 'fn_geo_err': []}
eval_smplx_dict = {'cont_precision': [], 'cont_recall': [], 'cont_f1': [], 'fp_geo_err': [], 'fn_geo_err': []}
eval_mano_dict = {'cont_precision': [], 'cont_recall': [], 'cont_f1': [], 'fp_geo_err': [], 'fn_geo_err': []}


eval_size = len(eval_outputs['img_path'])


for each_idx in tqdm(range(len(eval_outputs['img_path']))):
    img_path = eval_outputs['img_path'][each_idx:each_idx+1]


    # SMPL data
    contact_labels_3d_gt = eval_outputs['contact_labels_3d_gt'][each_idx:each_idx+1]
    contact_labels_3d_pred = eval_outputs['contact_labels_3d_pred'][each_idx:each_idx+1]

    # SMPLX data
    contact_labels_3d_smplx_gt = eval_outputs['contact_labels_3d_smplx_gt'][each_idx:each_idx+1]
    contact_labels_3d_smplx_pred = eval_outputs['contact_labels_3d_smplx_pred'][each_idx:each_idx+1]

    # MANO data
    contact_labels_3d_mano_gt = eval_outputs['contact_labels_3d_mano_gt'][each_idx:each_idx+1]
    contact_labels_3d_mano_pred = eval_outputs['contact_labels_3d_mano_pred'][each_idx:each_idx+1]

    contact_labels_3d_gt, contact_labels_3d_pred = torch.from_numpy(contact_labels_3d_gt), torch.from_numpy(contact_labels_3d_pred)
    contact_labels_3d_smplx_gt, contact_labels_3d_smplx_pred = torch.from_numpy(contact_labels_3d_smplx_gt), torch.from_numpy(contact_labels_3d_smplx_pred)
    contact_labels_3d_mano_gt, contact_labels_3d_mano_pred = torch.from_numpy(contact_labels_3d_mano_gt), torch.from_numpy(contact_labels_3d_mano_pred)


    # Eval on SMPL
    cont_smpl_pre, cont_smpl_rec, cont_smpl_f1 = precision_recall_f1score(contact_labels_3d_gt, contact_labels_3d_pred)
    # fp_geo_err, fn_geo_err = det_error_metric(contact_labels_3d_pred, contact_labels_3d_gt)
    eval_smpl_dict['cont_precision'].append(cont_smpl_pre.item())
    eval_smpl_dict['cont_recall'].append(cont_smpl_rec.item())
    eval_smpl_dict['cont_f1'].append(cont_smpl_f1.item())

    # Eval on SMPL-X
    cont_smplx_pre, cont_smplx_rec, cont_smplx_f1 = precision_recall_f1score(contact_labels_3d_smplx_gt, contact_labels_3d_smplx_pred)
    # fp_smplx_geo_err, fn_smplx_geo_err = det_error_metric(contact_labels_3d_smplx_pred, contact_labels_3d_smplx_gt) # TODO: Get distnace matrix (DIST_MATRIX) for SMPLX & MANO
    eval_smplx_dict['cont_precision'].append(cont_smplx_pre.item())
    eval_smplx_dict['cont_recall'].append(cont_smplx_rec.item())
    eval_smplx_dict['cont_f1'].append(cont_smplx_f1.item())

    # Eval on MANO
    cont_mano_pre, cont_mano_rec, cont_mano_f1 = precision_recall_f1score(contact_labels_3d_mano_gt, contact_labels_3d_mano_pred)
    # fp_mano_geo_err, fn_mano_geo_err = det_error_metric(contact_labels_3d_mano_pred, contact_labels_3d_mano_gt)
    eval_mano_dict['cont_precision'].append(cont_mano_pre.item())
    eval_mano_dict['cont_recall'].append(cont_mano_rec.item())
    eval_mano_dict['cont_f1'].append(cont_mano_f1.item())


# Calculate final evaluation

# SMPL
print('Test SMPL Contact Precision: ', np.sum(eval_smpl_dict['cont_precision'])/eval_size)
print('Test SMPL Contact Recall: ', np.sum(eval_smpl_dict['cont_recall'])/eval_size)
print('Test SMPL Contact F1 Score: ', np.sum(eval_smpl_dict['cont_f1'])/eval_size)

# SMPLX
print('Test SMPLX Contact Precision: ', np.sum(eval_smplx_dict['cont_precision'])/eval_size)
print('Test SMPLX Contact Recall: ', np.sum(eval_smplx_dict['cont_recall'])/eval_size)
print('Test SMPLX Contact F1 Score: ',np.sum( eval_smplx_dict['cont_f1'])/eval_size)

# MANO
print('Test MANO Contact Precision: ', np.sum(eval_mano_dict['cont_precision'])/eval_size)
print('Test MANO Contact Recall: ', np.sum(eval_mano_dict['cont_recall'])/eval_size)
print('Test MANO Contact F1 Score: ', np.sum(eval_mano_dict['cont_f1'])/eval_size)