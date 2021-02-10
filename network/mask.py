import numpy as np

"""
Ordering is an np.array with indices that one should
go through.
"""

def gen_raster_scan_order(L=3):
    """
    Generate raster scan ordering on a square lattice
    with size LxL.
    """
    return np.arange(L*L)

def gen_zigzag_order(L=3):
    """
    generate a zigzag ordering
    """
    lattice = np.arange(L*L).reshape([L, L])
    all_points = []

    for layer in range(2*L):
        # print("layer : ", layer)
        points_in_layer = []
        for idx_i in range(layer+1):
            idx_j = layer - idx_i
            if idx_i >= L or idx_j >= L:
                continue
            else:
                points_in_layer.append((idx_i, idx_j))

        all_points = all_points + points_in_layer[::(-1)**(layer+1)]
        # print(" all_points)

    return np.array([lattice[pt] for pt in all_points])

def gen_fc_mask_simple(ordering, mask_type, dtype=np.float64):
    """
    N = |ordering|
    Assume the maks is of size NxN.

    Type A mask connects all indices before the current index,
    not including current index.

    Type B mask connects all indices before the current index
    and the current index.
    """
    N = ordering.size
    mask = np.zeros([N, N], dtype=dtype)
    if mask_type == 'A':
        for count, block_idx in enumerate(ordering):
            mask[block_idx, ordering[:count]] = 1
    elif mask_type == 'B':
        for count, block_idx in enumerate(ordering):
            mask[block_idx, ordering[:count+1]] = 1
    else:
        raise NotImplementedError

    return mask.T  ## This is because the convention of x*W instead of W*x in tf implementation.

def gen_fc_mask(ordering, mask_type, dtype=np.float64,
                in_hidden=1, out_hidden=1):
    """
    N = |ordering|
    Assume the mask is of size (N*out_hidden) x (N*in_hidden).

    Type A mask connects all indices before the current index,
    not including current index.

    Type B mask connects all indices before the current index
    and the current index.
    """
    N = ordering.size
    mask = np.zeros([N * out_hidden, N * in_hidden], dtype=dtype)
    if mask_type == 'A':
        for count, block_idx in enumerate(ordering):
            for in_idx in ordering[:count]:
                mask[block_idx*out_hidden:(block_idx+1)*out_hidden,
                     in_idx*in_hidden:(in_idx+1)*in_hidden]  = 1
    elif mask_type == 'B':
        for count, block_idx in enumerate(ordering):
            for in_idx in ordering[:count+1]:
                mask[block_idx*out_hidden:(block_idx+1)*out_hidden,
                     in_idx*in_hidden:(in_idx+1)*in_hidden]  = 1
    else:
        raise NotImplementedError

    return mask.T  ## This is because the convention of x*W instead of W*x in tf implementation.

def gen_1d_conv_mask(mask_type, filter_size, in_ch, out_ch, dtype=np.float64):
    """
    generating a square mask with size filter_size.
    The shape is [filter_size, filter_size, in_ch, out_ch]
    """
    zeros_tensor = np.zeros((in_ch, out_ch), dtype=dtype)
    id_tensor = np.ones((in_ch, out_ch), dtype=dtype)
    assert filter_size % 2 == 1
    mask = np.zeros([filter_size, in_ch, out_ch], dtype=dtype)
    if mask_type == 'A':
        for j in range(filter_size//2):
            mask[j, :, :] = id_tensor

    elif mask_type == 'B':
        for j in range(filter_size//2+1):
            mask[j, :, :] = id_tensor

    elif mask_type == 'A2':
        mask = np.ones([filter_size, in_ch, out_ch], dtype=dtype)
        mask[-1, :, :] = np.zeros((in_ch, out_ch), dtype=dtype)
    else:
        raise NotImplementedError

    return mask

def gen_2d_conv_mask(mask_type, filter_size, in_ch, out_ch, dtype=np.float64):
    """
    generating a square mask with size filter_size.
    The shape is [filter_size, filter_size, in_ch, out_ch]
    """
    zeros_tensor = np.zeros((in_ch, out_ch), dtype=dtype)
    id_tensor = np.ones((in_ch, out_ch), dtype=dtype)
    assert filter_size % 2 == 1
    mask = np.zeros([filter_size, filter_size, in_ch, out_ch], dtype=dtype)
    if mask_type == 'A':
        for i in range(filter_size//2):
            for j in range(filter_size):
                mask[i,j,:,:] = id_tensor

        for j in range(filter_size//2):
            mask[filter_size//2,j,:,:] = id_tensor

    elif mask_type == 'B':
        for i in range(filter_size//2):
            for j in range(filter_size):
                mask[i,j,:,:] = id_tensor

        for j in range(filter_size//2+1):
            mask[filter_size//2,j,:,:] = id_tensor

    elif mask_type == 'A2':
        mask = np.ones([filter_size, filter_size, in_ch, out_ch], dtype=dtype)
        mask[-1, -1, :, :] = np.zeros((in_ch, out_ch), dtype=dtype)
    else:
        raise NotImplementedError

    return mask



if __name__ == '__main__':
    i = 3
    assert np.isclose(0, np.linalg.norm(np.array(gen_raster_scan_order(i))-np.arange(i**2)))
    assert np.isclose(0, np.linalg.norm(np.array(gen_zigzag_order(i))-
                                        np.array([0,1,3,6,4,2,5,7,8])))

    i = 4
    assert np.isclose(0, np.linalg.norm(np.array(gen_raster_scan_order(i))-np.arange(i**2)))
    assert np.isclose(0, np.linalg.norm(np.array(gen_zigzag_order(i))-
                                        np.array([0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15])))

    mask1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 0., 1., 1., 0., 1., 0., 0.],
                      [1., 1., 0., 0., 0., 0., 0., 0., 0.],
                      [1., 1., 0., 1., 0., 0., 1., 0., 0.],
                      [1., 1., 1., 1., 1., 0., 1., 0., 0.],
                      [1., 1., 0., 1., 0., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 0., 0.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 0.]])

    print("test pass!")


