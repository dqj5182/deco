# This script can be used to convert the contact labels from SMPL to SMPL-X format and vice-versa.

import os
import argparse
import pickle as pkl
import torch
import numpy as np
from common import constants

def convert_contacts(contact_data, mapping, in_key, out_key):
    """
    Converts the contact labels from SMPL to SMPL-X format and vice-versa.

    Args:
        contact_labels: contact labels in SMPL or SMPL-X format
        mapping: mapping from SMPL to SMPL-X vertices or vice-versa

    Returns:
        contact_labels_converted: converted contact labels
    """
    contact_labels = contact_data[in_key]

    if not isinstance(contact_labels, torch.Tensor):
        contact_labels = torch.from_numpy(contact_labels).float()
    if not isinstance(mapping, torch.Tensor):
        mapping = torch.from_numpy(mapping).float()

    bs = contact_labels.shape[0]
    mapping = mapping[None].expand(bs, -1, -1)
    contact_labels_converted = torch.bmm(mapping, contact_labels[..., None])
    contact_labels_converted = contact_labels_converted.squeeze()

    contact_data[out_key] = contact_labels_converted.numpy()

    return contact_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact_npz', type=str, required=True, help='path to contact npz file',
                        default='../datasets/ReleaseDatasets/damon/hot_dca_train.npz')
    parser.add_argument('--input_type', type=str, required=True, help='input type: smpl or smplx',
                        default='smpl')
    args = parser.parse_args()
    

    # Load mapping between smpl and smplx vertices
    if args.input_type == 'smpl':
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smpl_to_smplx.pkl")
    elif args.input_type == 'smplx':
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smplx_to_smpl.pkl")
    else:
        raise ValueError('input_type must be smpl or smplx')
    
    with open(mapping_pkl, 'rb') as f:
        mapping = pkl.load(f)
        mapping = mapping["matrix"]


    # Get contact labels
    contact_data = np.load(args.contact_npz, allow_pickle=True)
    contact_data = dict(contact_data)
    
    # contact_data = convert_contacts(contact_data, mapping, 'contact_label', 'contact_label_smplx') # -> Original for converting file "hot_dca_trainval.npz"
    if args.input_type == 'smpl':
        contact_data = convert_contacts(contact_data, mapping, 'contact_labels_3d_gt', 'contact_labels_3d_smplx_gt')
        contact_data = convert_contacts(contact_data, mapping, 'contact_labels_3d_pred', 'contact_labels_3d_smplx_pred')
    else:
        import pdb; pdb.set_trace()

    # save the converted contact labels
    parent_dir, child_dir = os.path.split(args.contact_npz)
    child_name, child_ext = os.path.splitext(child_dir)

    if args.input_type == 'smpl':
        save_contact_path = os.path.join(parent_dir, f'{child_name}_smplx{child_ext}')
    elif args.input_type == 'smplx':
        save_contact_path = os.path.join(parent_dir, f'{child_name}_smpl{child_ext}')
    else:
        raise ValueError('input_type must be smpl or smplx')
    

    np.savez(save_contact_path, **contact_data)



