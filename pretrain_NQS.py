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
    using_complex = args.using_complex
    opt = args.opt  # "Mom"

    system_size = (L, 2)
    Net = tf_network(which_net, system_size, optimizer=opt,
                     dim=1, alpha=alpha, activation=act,
                     using_complex=using_complex
                    )

    X_computation_basis = np.genfromtxt('ExactDiag/basis_L%d.csv' % L, delimiter=',')
    X = np.zeros([2**L, L, 2])
    X[:,:,0] = X_computation_basis.reshape([2**L, L])
    X[:,:,1] = 1-X_computation_basis.reshape([2**L, L])
    Y = np.genfromtxt('ExactDiag/EigVec/ES_L'+str(L)+'_J2_'+str(int(J2*10))+'_OBC.csv').reshape((2**L, 1))

    print(X.shape, Y.shape)

    with Net.sess as sess:

        true_amp = tf.placeholder(tf.complex64, [None, 1])
        v1 = true_amp
        v2 = Net.amp
        ###################################
        # Cost function 1: Batch fidelity #
        ###################################
        cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)

        ###################################
        # KL-divergence + classficiation error
        ###################################
        # cost1 = - tf.log(tf.divide(v2**2+1e-60, v1**2 + 1e-60))
        # cost2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sign(v1), logits=v2)
        # cost = tf.reduce_sum(cost1 + cost2)

        # cost = tf.reduce_sum( tf.multiply(tf.log(tf.divide(v2, v1)), tf.log(tf.divide(v1, v2))) )
        # cost = -tf.real(tf.norm( tf.log(tf.complex(v2,0.))-tf.log(tf.complex(v1,0.)) ))
        # cost = -tf.reduce_sum(tf.divide(v2,v1+1e-8)) + 1. * tf.norm(v2)

        # cost = -tf.reduce_sum(tf.multiply(true_amp, tf.log(Net.amp)))
        # cost = tf.nn.l2_loss((Net.amp - true_amp))
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
        print(("Total num para: ", Net.getNumPara()))

        p = (np.abs(Y)**2).flatten()
        # p = p + 100./batch_size
        # p = p/sum(p)

        total_cos_accu = []
        batch_cos_accu = []

        for i in range(100000+1):
            batch_mask = np.random.choice(len(Y), batch_size, p=p)

            if i % 5000 == 0:
                ### get full batch information
                y = Net.get_amp(X)
                print(('y norm : ', np.linalg.norm(y)))
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

            _, c, y = sess.run([train_step, cost, Net.amp], feed_dict={Net.x: X[batch_mask],
                                                                       true_amp: Y[batch_mask]})
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
