import time
import numpy as np

from hoshen_kopelman import label, label_parallel
from hoshen_kopelman_py import label as label_py # label_parallel

if __name__ == "__main__":


    # print(ferro_board)
    # result = label(ferro_board)
    # print('final_board:\n', result[0])
    # print('label size:\n', [len(l) for l in result[1]])
    # print(correct_label)

    num_iter = 100
    num_data = 500
    lattice_size = 20
    array_shape = (num_data, lattice_size, lattice_size)

    ferro_board = np.random.randint(2, size=[num_data, lattice_size, lattice_size], dtype=np.int32)

    # result = np.zeros([num_data, lattice_size, lattice_size])
    # result2 = np.zeros([num_data, lattice_size, lattice_size])
    '''
    result = label(ferro_board)
    print('board:')
    print(ferro_board)
    print('result:')
    print(result)
    '''

#     start_c, start_t = time.clock(), time.time()
#     for k in range(num_iter):
#         result=label_py(ferro_board)
# 
#     end_c, end_t = time.clock(), time.time()
#     print('py time : ',end_c - start_c, end_t - start_t)
# 
# 
    start_c, start_t = time.clock(), time.time()
    for k in range(num_iter):
        result2 = label(ferro_board)

    end_c, end_t = time.clock(), time.time()
    print('serial time : ',end_c - start_c, end_t - start_t)
#     print('difference norm : ', np.linalg.norm(np.array(result)-np.array(result2)))


    start_c, start_t = time.clock(), time.time()
    for k in range(num_iter):
        result = label_parallel(ferro_board)

    end_c, end_t = time.clock(), time.time()
    print('parallel time : ',end_c - start_c, end_t - start_t)
    print('difference norm : ', np.linalg.norm(np.array(result)-np.array(result2)))


