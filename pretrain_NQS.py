from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
# from utils.prepare_net import prepare_net
from network.tf_network import tf_network
import network.tf_wrapper as tf_


if __name__ == "__main__":

    args = parse_args()
    L = args.L
    which_net = args.which_net
    lr = args.lr
    batch_size = args.batch_size

    alpha_map = {"NN": 2, "NN3": 2, "NN_complex": 1, "NN3_complex": 2,
                 "NN_RBM": 2}
    if args.alpha != 0:
        alpha = args.alpha
    else:
        alpha = alpha_map[which_net]

    opt = args.opt  # "Mom"
    system_size = (L, 2)
    # Net = prepare_net(which_net, system_size, opt, alpha)
    Net = tf_network(which_net, system_size, optimizer=opt, alpha=alpha)

    basis = []
    for line in open('EigenVec/basisMatrix'+str(L)+'.csv', 'r'):
        basis.append(line[:-1])

    newbasis = np.zeros((len(basis), L, 2))
    for i in range(len(basis)):
        for j in range(L):
            newbasis[i, j, 0] = basis[i][j]

    num_train = 2**L
    to_large_dict = {}
    X = np.zeros((num_train, L, 2))
    for i in range(0, num_train):
        temp = i+1
        for j in range(L):
            if temp > 2**(L-j-1):
                X[i, j, 0] = 1
                temp -= 2**(L-j-1)
            else:
                X[i, j, 1] = 1
                pass
        to_large_dict[''.join([str(int(ele)) for ele in X[i, :, 0]])] = i

    # X = X.reshape((num_train, L*2))

    # Y = np.zeros((num_train, 1))
    # amp_array = np.genfromtxt(open('EigenVec/EigVec_L'+str(L)+'V20W0E0.csv', 'r'))
    # for i in range(len(amp_array)):
    #     idx = to_large_dict[''.join([str(int(ele)) for ele in newbasis[i, :, 0]])]
    #     Y[idx] = amp_array[i]

    # X_nz = X[(np.abs(Y) * np.ones((1, 32)) >= 1e-15)]
    # Y_nz = Y[(np.abs(Y) >= 1e-15)]
    # X = X_nz.reshape((len(Y_nz), 32))
    # Y = Y_nz.reshape((len(Y_nz),1))

    # y for all Sz sector #
    Y = np.genfromtxt('EigenVec/eig_L'+str(L)+'_PBC.csv').reshape((2**L, 1))
    print X.shape, Y.shape

    with Net.sess as sess:

        true_out = tf.placeholder(tf.float32, [None, 1])
        v1 = true_out
        v2 = Net.pred
        cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)
        # cost = -tf.reduce_sum(tf.multiply(true_out, tf.log(Net.pred)))
        # cost = tf.nn.l2_loss((Net.pred - true_out))

        learning_rate = tf.Variable(lr)
        Optimizer = tf_.select_optimizer(optimizer=opt, learning_rate=learning_rate,
                                         momentum=0.9)
        train_step = Optimizer.minimize(cost)

        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())

        print len(Net.model_var_list), len(Net.para_list)
        saver = tf.train.Saver(Net.model_var_list)  # Net.model_var_list)
        ckpt_path = 'wavefunction/Pretrain/'+which_net+'/L'+str(L)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(Net.sess, ckpt.model_checkpoint_path)
            print("Restore from last check point")
        else:
            print("No checkpoint found")

        #    sess.run(learning_rate.assign(0.1 * (0.8**(i/10000))))
        print("-------- Start training -------\n")
        print("Total num para: ", Net.getNumPara())

        p = (np.abs(Y)).flatten()
        p = p + 0.1/batch_size
        p = p/sum(p)

        total_cos_accu = []
        batch_cos_accu = []
        # import pdb;pdb.set_trace()

        for i in xrange(100000+1):
            # batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
            batch_mask = np.random.choice(len(Y), batch_size, p=p)

            if i % 5000 == 0:
                y = sess.run(Net.pred, feed_dict={Net.x: X})
                print('y norm : ', np.linalg.norm(y))
                c = Y.flatten().dot(y.flatten())/np.linalg.norm(Y)/np.linalg.norm(y)
                print(c)
                total_cos_accu.append(c)
                PLOT = True
                if PLOT:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(Y/np.linalg.norm(Y), '-o')
                    plt.plot(y/np.linalg.norm(y), '--')
                    plt.show()
                pass

            _, c, y = sess.run([train_step, cost, Net.pred], feed_dict={Net.x: X[batch_mask],
                                                                        true_out: Y[batch_mask]})
            batch_cos_accu.append(-c)

            if i % 500 == 0:
                print("step:", i, " cosine accuracy:", -c, "Y norm",
                      np.linalg.norm(Y[batch_mask]), "y norm:", np.linalg.norm(y))
# "dot:",
# y.T.dot(Y[batch_mask])[0]/np.linalg.norm(Y[batch_mask])/np.linalg.norm(y))
                saver.save(sess, ckpt_path + '/pre')

    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_batch.csv' % (L, which_net, alpha, opt, lr),
               batch_cos_accu, '%.4e', delimiter=',')
    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_total.csv' % (L, which_net, alpha, opt, lr),
               total_cos_accu, '%.4e', delimiter=',')

    # fig.savefig('L16_pretrain.eps',bbox_inches='tight')
