import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
from network.tf_network import tf_network
import network.tf_wrapper as tf_


if __name__ == "__main__":

    args = parse_args()
    L = args.L
    which_net = args.which_net
    lr = args.lr
    batch_size = args.batch_size
    J2 = args.J2
    alpha = args.alpha
    reg = args.reg
    act = args.act

    opt = args.opt  # "Mom"
    system_size = (L, L, 2)
    Net = tf_network(which_net, system_size, optimizer=opt, dim=2, alpha=alpha, activation=act)

    X_half = np.genfromtxt('X2.csv', delimiter=',').reshape((60000, 8, 8))
    X = np.zeros((60000,8,8,2))
    X[:,:,:,0] = X_half
    X[:,:,:,1] = 1-X_half
    Y = np.genfromtxt('Y2.csv', delimiter=',').reshape((60000, 1))
    # Y = np.sign(Y)

    X_half = np.genfromtxt('X16.csv', delimiter=',')
    X = np.zeros([2**16, 4, 4, 2])
    X[:,:,:,0] = X_half.reshape([2**16, 4, 4])
    X[:,:,:,1] = 1-X_half.reshape([2**16, 4, 4])
    Y = np.genfromtxt('ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')

    print(X.shape, Y.shape)

    with Net.sess as sess:

        true_out = tf.placeholder(tf.float32, [None, 1])
        v1 = true_out
        v2 = tf.real(Net.amp)
        cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)
        # cost = -tf.reduce_sum(tf.multiply(true_out, tf.log(Net.amp)))
        # cost = tf.nn.l2_loss((Net.amp - true_out))
        for w in Net.para_list:
            cost += reg * tf.nn.l2_loss(w)


        learning_rate = tf.Variable(lr)
        Optimizer = tf_.select_optimizer(optimizer=opt, learning_rate=learning_rate,
                                         momentum=0.9)
        train_step = Optimizer.minimize(cost)

        # from tensorflow.python import debug as tf_debug
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        sess.run(tf.global_variables_initializer())

        print(len(Net.model_var_list), len(Net.para_list))
        saver = tf.train.Saver(Net.model_var_list)
        # ckpt_path = 'wavefunction/Pretrain/'+which_net+'/L'+str(L)
        ckpt_path = 'wavefunction/vmc2d/'+which_net+'_'+act+'/L'+str(L)+'a'+str(alpha)
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
        print(("Total num para: ", Net.getNumPara()))

        p = (np.abs(Y)).flatten()
        p = p/sum(p)
        p = p + 100./batch_size
        p = p/sum(p)
        Y = Y/np.sum(np.abs(Y))

        total_cos_accu = []
        batch_cos_accu = []
        # import pdb;pdb.set_trace()

        for i in range(100000):
            # batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
            batch_mask = np.random.choice(len(Y), batch_size, p=p)

            if i % 5000 == 0:
                # y = sess.run(Net.amp, feed_dict={Net.x: X})
                y = Net.get_amp(X)
                print(('y norm : ', np.linalg.norm(y)))
                y_list = []
                for i in range(60):
                    yi = Net.get_amp(X[i*1000:(i+1)*1000])
                    y_list.append(yi)

                y_list.append(Net.get_amp(X[60000:]))
                y2 = np.concatenate(y_list)
                print(('y2 norm : ', np.linalg.norm(y2)))
                print(('diff in y-y2', np.linalg.norm(y-y2)))

                c = Y.flatten().dot(y.flatten())/np.linalg.norm(Y)/np.linalg.norm(y)
                print(c)
                total_cos_accu.append(c)
                PLOT = True
                import pdb;pdb.set_trace()
                mask = np.sum(X[:,:,:,0], axis=(1,2)) == 8
                y_mask = y[mask]
                print("Sz 0 prob : ", y_mask.T.conjugate().dot(y_mask))
                pdb.set_trace()
                if PLOT:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(Y/np.linalg.norm(Y), '-o')
                    plt.plot(y.real/np.linalg.norm(y.real), '--')
                    plt.show()
                pass

            _, c, y = sess.run([train_step, cost, Net.amp], feed_dict={Net.x: X[batch_mask],
                                                                       true_out: Y[batch_mask]})
            batch_cos_accu.append(-c)

            if i % 500 == 0:
                print(("step:", i, " cosine accuracy:", -c, "Y norm",
                      np.linalg.norm(Y[batch_mask]), "y norm:", np.linalg.norm(y)))
# "dot:",
# y.T.dot(Y[batch_mask])[0]/np.linalg.norm(Y[batch_mask])/np.linalg.norm(y))
                saver.save(sess, ckpt_path + '/pre')

    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_batch.csv' % (L, which_net, alpha, opt, lr),
               batch_cos_accu, '%.4e', delimiter=',')
    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_total.csv' % (L, which_net, alpha, opt, lr),
               total_cos_accu, '%.4e', delimiter=',')

    # fig.savefig('L16_pretrain.eps',bbox_inches='tight')
