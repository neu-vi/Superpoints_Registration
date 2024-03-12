import torch
import MinkowskiEngine as ME

def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """
    batch_sz = len(list_data)

    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    to_retain_as_list = ['src_xyz', 'tgt_xyz', 'tgt_raw',
                         'src_overlap', 'tgt_overlap',
                         'correspondences',
                         'src_path', 'tgt_path',
                         'idx']

    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    
    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data

def collate_tensors(list_data):
    """
    Collates the modelnet dataset into a stack of tensors since each pointcloud in modelnet is of the same size
    """

    batch_sz = len(list_data)

    to_retain_as_list = []
    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    
    data['src_xyz'] =  torch.stack([list_data[b]['src_xyz'].T for b in range(batch_sz)], dim=0)
    data['tgt_xyz'] =  torch.stack([list_data[b]['tgt_xyz'].T for b in range(batch_sz)], dim=0)
    data['tgt_raw'] =  torch.stack([list_data[b]['tgt_raw'].T for b in range(batch_sz)], dim=0)


    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data

def collate_sparse_tensors(list_data):
    batch_sz = len(list_data)
    data = {}
    coords_src = [list_data[b]['coords_src'] for b in range(batch_sz)]
    feats_src = [list_data[b]['feats_src'] for b in range(batch_sz)]
    coords_tgt = [list_data[b]['coords_tgt'] for b in range(batch_sz)]
    feats_tgt = [list_data[b]['feats_tgt'] for b in range(batch_sz)]

    data['coords_src'], data['feats_src'] = ME.utils.sparse_collate(coords=coords_src, feats=feats_src)
    data['coords_tgt'], data['feats_tgt'] = ME.utils.sparse_collate(coords=coords_tgt, feats=feats_tgt)

    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    
    data['src_xyz'] =  torch.stack([list_data[b]['src_xyz'] for b in range(batch_sz)], dim=0)
    data['tgt_xyz'] =  torch.stack([list_data[b]['tgt_xyz'] for b in range(batch_sz)], dim=0)
    data['tgt_raw'] =  torch.stack([list_data[b]['tgt_raw'] for b in range(batch_sz)], dim=0)

    return data
