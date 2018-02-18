import time
import numpy as np
# from numba import jit, autojit
from multiprocessing import Pool

''''
Example as below,
Input:

ferro_board = np.array([[1,0,0,0,1,1],
                        [1,1,0,0,0,1],
                        [0,0,1,1,0,1],
                        [0,1,0,1,0,1],
                        [0,1,1,0,1,0],
                        [1,0,1,0,0,1]], dtype=np.int)

Should have the output as :
correct_label = np.array([[10,  -12,-12,    -12,    10,     10],
                          [10,  10, -12,    -12,    -12,    10],
                          [-5,   -5,  3,    3,      -12,    10],
                          [-5,   4,  -1,    3,      -12,    10],
                          [-5,   4,  4,     -12,   1,      -5],
                          [10,  -12, 4,  -12,  -12,  10]])

'''

# @jit('float_[:,:](float_[:,:])')
def label(config_array, cutoff=5):
    '''
    Input array of configurations.
    Input type: nd-array
    Input shape: [num_data, Lx, Ly]
    Output Configuration with cluster size labeling.
    Output type: nd-array
    Output shape: [num_data, Lx, Ly]
    '''
    input_shape = np.shape(config_array)
    labeled_config_array = np.zeros(input_shape, dtype=np.int)
    num_data = input_shape[0]
    Lx = input_shape[1]
    Ly = input_shape[2]

    for idx_data in range(num_data):
        board = config_array[idx_data,:,:]
        label_board = np.zeros((Lx, Ly), dtype=np.int)
        max_label=0
        label_site = [[]]

        for i in range(Lx):
            for j in range(Ly):
                site = (i,j)
                if board[site] == 0:
                    continue
                left_site = ((i+Lx-1)%Lx, j)
                up_site = (i, (j+Ly-1)%Ly)
                left_label = label_board[left_site]
                up_label = label_board[up_site]
                if (left_label == 0 and up_label == 0):
                    max_label = max_label + 1
                    label_site.append([])
                    label_board[site] = max_label
                    label_site[max_label].append(site)
                elif (left_label != 0 and up_label == 0):
                    label_board[site] = left_label
                    label_site[left_label].append(site)
                elif (left_label == 0 and up_label != 0):
                    label_board[site] = up_label
                    label_site[up_label].append(site)
                else:  # elif (left_label != 0 and up_label != 0):
                    if left_label == up_label:
                        label_board[site] = up_label
                        label_site[up_label].append(site)
                        continue
                    elif left_label > up_label:
                        small_label, large_label = up_label, left_label
                    else:
                        small_label, large_label = left_label, up_label

                    label_board[site] = small_label
                    label_site[small_label].append(site)
                    for iter_site in label_site[large_label]:
                        label_board[iter_site] = small_label

                    label_site[small_label] = label_site[small_label]+label_site[large_label][:]
                    label_site[large_label] = []
                    pass # Union

        for i in range(Lx):
            j=0
            site = (i,j)
            if board[site] == 0:
                continue
            left_site = ((i+Lx-1)%Lx, j)
            up_site = (i, (j+Ly-1)%Ly)
            my_label = label_board[site]
            up_label = label_board[up_site]
            if (my_label != 0 and up_label != 0):
                if my_label == up_label:
                    continue
                elif my_label > up_label:
                    small_label, large_label = up_label, my_label
                else:
                    small_label, large_label = my_label, up_label

                for site in label_site[large_label]:
                    label_board[site] = small_label

                label_site[small_label] = label_site[small_label]+label_site[large_label][:]
                label_site[large_label] = []
                # label_site.pop(left_label)
                pass # Union
            else:
                pass

        for j in range(Ly):
            i=0
            site = (i,j)
            if board[site] == 0:
                continue
            left_site = ((i+Lx-1)%Lx, j)
            left_label = label_board[left_site]
            my_label = label_board[site]
            if (left_label != 0 and my_label != 0):
                if my_label == left_label:
                    continue
                elif left_label > my_label:
                    small_label, large_label = my_label, left_label
                else:
                    small_label, large_label = left_label, my_label

                for site in label_site[large_label]:
                    label_board[site] = small_label

                label_site[small_label] = label_site[small_label]+label_site[large_label][:]
                label_site[large_label] = []
                # label_site.pop(left_label)
                pass # Union
            else:
                pass

            for idx_label in range(len(label_site)):
                cluster_size = len(label_site[idx_label])
                if cluster_size > cutoff:
                    cluster_size = cutoff
                for site in label_site[idx_label]:
                    labeled_config_array[idx_data, site[0], site[1]] = cluster_size

        # labeled_config_array[idx_data,:,:] = label_board


    return labeled_config_array # , label_site

# print(ferro_board)
# result = label(ferro_board)
# print('final_board:\n', result[0])
# print('label size:\n', [len(l) for l in result[1]])
# print(correct_label)


num_data = 2
ferro_board = np.random.randint(2,size=[num_data,6, 6])
result = np.zeros([num_data,6, 6])
result2 = np.zeros([num_data,6, 6])

result = label(ferro_board)
print('board:')
print(ferro_board)
print('result:')
print(result)

'''
start_c, start_t = time.clock(), time.time()
agents = 4
chunksize = num_data // agents
for k in range(100):
    with Pool(processes=agents) as pool:
        result = pool.map(label, ferro_board, chunksize)
        pool.close()

end_c, end_t = time.clock(), time.time()
print('parallel time : ',end_c - start_c, end_t - start_t)
'''

'''
start_c, start_t = time.clock(), time.time()
for k in range(100):
    for idx in range(num_data):
        # print(ferro_board)
        result2[idx] = label(ferro_board[idx])
        # print('final_board:\n', result[0])
        # print('label size:\n', [len(l) for l in result[1]])
        # print(correct_label)

end_c, end_t = time.clock(), time.time()
print('serial time : ',end_c - start_c, end_t - start_t)
print('difference norm : ', np.linalg.norm(np.array(result)-np.array(result2)))
'''


