from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
from network.tf_network import tf_network
import network.tf_wrapper as tf_


if __name__ == "__main__":

    args = parse_args()
    (L, which_net, lr, num_sample, J2) = (args.L, args.which_net, args.lr, args.num_sample, args.J2)
    alpha = args.alpha
    opt, batch_size, H, dim, num_iter  = (args.opt, args.batch_size,
                                          args.H, args.dim, args.num_iter)
    if dim == 1:
        systemSize = (L, 2)
        N = L
    elif dim == 2:
        systemSize = (L, L, 2)
        N = L*L
    else:
        raise NotImplementedError

    batch_size = 100
    Net = tf_network(which_net, systemSize, optimizer='Mom', dim=dim, alpha=alpha)
    print("Total num para: ", Net.getNumPara())


    '''
    Create configuration, i.e. product state basis, for 1d lattice problem
    '''
    # total dimension of the Hilbert Space
    full_dim = 2**N
    # Create a mapping between subspace indices and full vectorspace indices
    to_large_dict = {}
    # Basis for full vectorspace
    X = np.zeros((full_dim, N, 2))
    for i in range(0, full_dim):
        temp = i+1
        for j in range(N):
            if temp > 2**(N-j-1):
                X[i, j, 0] = 1
                temp -= 2**(N-j-1)
            else:
                X[i, j, 1] = 1
                pass
        to_large_dict[''.join([str(int(ele)) for ele in X[i, :, 0]])] = i

    # coefficient of the full vector
    Y = np.genfromtxt('ExactDiag/EigVec/ES_%dd_L%dx%d_J2_%d.csv' %(dim, L, L, J2*10))
    # Y = np.genfromtxt('ExactDiag/EigVec/ES_L16_J2_%d.csv' % J2).reshape((2**L, 1))
    print(X.shape, Y.shape)
    # create a mask to restrict to the subspace
    mask = (np.einsum('ij->i',X[:,:,0])==8)
    # X now is basis of the restricted subspace
    # Y, the coefficient of the vecotr in restricted subspace
    X = X[mask, :,:]
    Y = Y[mask]

    '''
    reshape the 1d lattice to 2d lattice
    '''
    if dim==2:
        X = np.reshape(X, [X.shape[0], 4, 4, 2])
        Y = np.reshape(Y, [Y.shape[0], 1])

    print(X.shape, Y.shape)


    fidelity_list = []
    with Net.sess as sess:
        for idx in [3,15,18,14,10
                   ]:# range(1, 101):

            true_out = tf.placeholder(tf.float32, [None, 1])
            v1 = true_out
            v2 = Net.pred
            cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)
            # cost = -tf.reduce_sum(tf.multiply(true_out, tf.log(Net.pred)))
            # cost = tf.nn.l2_loss((Net.pred - true_out))

            # learning_rate = tf.Variable(lr)
            # Optimizer = tf_.select_optimizer(optimizer=opt, learning_rate=learning_rate,
            #                                  momentum=0.9)
            # train_step = Optimizer.minimize(cost) 

            sess.run(tf.global_variables_initializer())

            print(len(Net.model_var_list), len(Net.para_list))
            saver = tf.train.Saver(Net.model_var_list)
            ckpt_path = '../Job_Result/2d_sRBM_a16/731106/731106.%d/wavefunction/vmc2d/sRBM/L4a16/' % (idx)
            # ckpt_path = '../Job_Result/2d_sRBM/727101/727101.%d/wavefunction/vmc2d/sRBM/L4a4/' % (idx)
            # ckpt_path = '../Job_Result/J1J2.%d/L16_sRBM_a4_Mom1e-03_S1000/%d/wavefunction/vmc1d/sRBM/L16a4/' % (J2, idx)
            if not os.path.exists(ckpt_path):
                raise
                os.makedirs(ckpt_path)
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(Net.sess, ckpt.model_checkpoint_path)
                print("Restore from last check point")
            else:
                print("No checkpoint found")

            p = (np.abs(Y)).flatten()
            p = p + 0.1/batch_size
            p = p/sum(p)

            total_cos_accu = []
            batch_cos_accu = []
            # import pdb;pdb.set_trace()

            for i in range(0+1):
                # batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
                batch_mask = np.random.choice(len(Y), batch_size, p=p)

                if i % 5000 == 0:
                    y = sess.run(Net.pred, feed_dict={Net.x: X})
                    print('y norm : ', np.linalg.norm(y))
                    c = Y.flatten().dot(y.flatten())/np.linalg.norm(Y)/np.linalg.norm(y)
                    print(c)
                    fidelity_list.append(c)
                    PLOT = True
                    if PLOT:
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        plt.plot(Y/np.linalg.norm(Y), '-o')
                        plt.plot(y/np.linalg.norm(y), '--')
                        plt.show()
                    pass

                c, y = sess.run([cost, Net.pred], feed_dict={Net.x: X[batch_mask],
                                                             true_out: Y[batch_mask]})
                print(c)
                batch_cos_accu.append(-c)

        # np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_batch.csv' % (L, which_net, alpha, opt, lr),
        #            batch_cos_accu, '%.4e', delimiter=',')
        # np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_total.csv' % (L, which_net, alpha, opt, lr),
        #            total_cos_accu, '%.4e', delimiter=',')

        # fig.savefig('L16_pretrain.eps',bbox_inches='tight')

    print(np.sort(fidelity_list))
