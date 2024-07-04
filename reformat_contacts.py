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
    parser.add_argument('--pred_output', action="store_true")
    parser.add_argument('--only_get_mano', action="store_true")
    parser.add_argument('--only_mano_two_hands', action="store_true")
    parser.add_argument('--flip_left_to_right', action="store_true")
    args = parser.parse_args()
    

    # Load mapping between smpl and smplx vertices
    num_smplx_verts = 10475
    num_smpl_verts = 6890
    num_mano_verts = 778

    if args.input_type == 'smpl':
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smpl_to_smplx.pkl")
    elif args.input_type == 'smplx':
        mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smplx_to_smpl.pkl")
    else:
        raise ValueError('input_type must be smpl or smplx')
    
    with open(mapping_pkl, 'rb') as f:
        mapping = pkl.load(f)
        mapping = mapping["matrix"]


    # SMPLX-to-MANO mapping
    smplx_mano_mapping_pkl = os.path.join(constants.CONTACT_MAPPING_PATH, "smplx_to_mano.pkl") # from MANO_SMPLX_vertex_ids.pkl by SMPLX
    with open(smplx_mano_mapping_pkl, 'rb') as f:
        mano_mapping = pkl.load(f)
        mano_mapping_r = mano_mapping["right_hand"]
        mano_mapping_l = mano_mapping["left_hand"]


    # Get contact labels
    contact_data = np.load(args.contact_npz, allow_pickle=True)
    contact_data = dict(contact_data)
    

    # contact_data = convert_contacts(contact_data, mapping, 'contact_label', 'contact_label_smplx') # -> Original for converting file "hot_dca_trainval.npz"
    if args.input_type == 'smpl':
        if 'contact_label_smplx' not in [*contact_data]:
            contact_data = convert_contacts(contact_data, mapping, 'contact_labels_3d_gt', 'contact_labels_3d_smplx_gt')
        if args.pred_output:
            contact_data = convert_contacts(contact_data, mapping, 'contact_labels_3d_pred', 'contact_labels_3d_smplx_pred')

        # Save hand contact in MANO
        # Right hand
        if 'contact_label_smplx' not in [*contact_data]:
            contact_data['contact_labels_3d_mano_r_gt'] = contact_data['contact_labels_3d_smplx_gt'][:, mano_mapping_r]
        else:
            contact_data['contact_labels_3d_mano_r_gt'] = contact_data['contact_label_smplx'][:, mano_mapping_r]
        if args.pred_output:
            contact_data['contact_labels_3d_mano_r_pred'] = contact_data['contact_labels_3d_smplx_pred'][:, mano_mapping_r]

        # Left hand
        if 'contact_label_smplx' not in [*contact_data]:
            contact_data['contact_labels_3d_mano_l_gt'] = contact_data['contact_labels_3d_smplx_gt'][:, mano_mapping_l]
        else:
            contact_data['contact_labels_3d_mano_l_gt'] = contact_data['contact_label_smplx'][:, mano_mapping_l]
        if args.pred_output:
            contact_data['contact_labels_3d_mano_l_pred'] = contact_data['contact_labels_3d_smplx_pred'][:, mano_mapping_l]

        # All hand
        contact_data['contact_labels_3d_mano_gt'] = np.concatenate((contact_data['contact_labels_3d_mano_r_gt'], contact_data['contact_labels_3d_mano_l_gt']), axis=-1)
        if args.pred_output:
            contact_data['contact_labels_3d_mano_pred'] = np.concatenate((contact_data['contact_labels_3d_mano_r_pred'], contact_data['contact_labels_3d_mano_l_pred']), axis=-1)


        # If you want to only save mano as main contact labels
        if args.only_get_mano:
            if args.only_mano_two_hands:
                contact_data['contact_label'] = contact_data['contact_labels_3d_mano_gt']
            else:
                contact_data['contact_label'] = contact_data['contact_labels_3d_mano_r_gt']
    else:
        import pdb; pdb.set_trace()


    # Flip left hand data as right hand data
    new_contact_data = {'is_right': []}

    if args.flip_left_to_right:
        for hand_type in range(2):
            # for each_idx in range(len(contact_data['imgname'])):
            for each_key in [*contact_data]:
                # First time of the key
                if each_key not in [*new_contact_data]:
                    new_contact_data[each_key] = []
                
                if hand_type == 0: # left hand
                    if each_key in ['contact_label']:
                        new_contact_data['contact_label'].extend(contact_data['contact_labels_3d_mano_l_gt'])
                        new_contact_data['is_right'].extend([0] * len(contact_data['imgname']))
                    else:
                        new_contact_data[each_key].extend(contact_data[each_key].tolist())
                elif hand_type == 1: # right hand
                    if each_key in ['contact_label']:
                        new_contact_data['contact_label'].extend(contact_data['contact_labels_3d_mano_r_gt'])
                        new_contact_data['is_right'].extend([1] * len(contact_data['imgname']))
                    else:
                        new_contact_data[each_key].extend(contact_data[each_key].tolist())
                else:
                    import pdb; pdb.set_trace()

                

        for each_key in [*new_contact_data]:
            new_contact_data[each_key] = np.array(new_contact_data[each_key])
        

        contact_data = new_contact_data


    # Save the converted contact labels
    parent_dir, child_dir = os.path.split(args.contact_npz)
    child_name, child_ext = os.path.splitext(child_dir)

    if args.input_type == 'smpl':
        if args.only_get_mano:
            save_contact_path = os.path.join(parent_dir, f'{child_name}_mano_r{child_ext}')
        else:
            save_contact_path = os.path.join(parent_dir, f'{child_name}_smplx{child_ext}')
    elif args.input_type == 'smplx':
        save_contact_path = os.path.join(parent_dir, f'{child_name}_smpl{child_ext}')
    else:
        raise ValueError('input_type must be smpl or smplx')
    

    np.savez(save_contact_path, **contact_data)