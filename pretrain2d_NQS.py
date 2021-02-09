import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
from network.tf_network import tf_network
import network.tf_wrapper as tf_


if __name__ == "__main__":

    args = parse_args()
    if bool(args.debug):
        np.random.seed(0)
        tf.set_random_seed(1234)
    else:
        pass

    (L, which_net, lr, num_sample) = (
        args.L, args.which_net, args.lr, args.num_sample)
    (J2, SR, reg, path) = (args.J2, bool(args.SR), args.reg, args.path)
    (act, SP, using_complex) = (args.act, bool(args.SP), bool(args.using_complex))
    (real_time, integration, pinv_rcond) = (
        bool(args.real_time), args.integration, args.pinv_rcond)

    if len(path) > 0 and path[-1] != '/':
        path = path + '/'

    if args.alpha != 0:
        alpha = args.alpha
    else:
        alpha = alpha_map[which_net]

    opt, batch_size, H, dim, num_iter = (args.opt, args.batch_size,
                                         args.H, args.dim, args.num_iter)
    PBC = args.PBC

    num_blocks, multi_gpus = args.num_blocks, args.multi_gpus
    save_each = args.save_each
    conserved_Sz, warm_up, Q_tar = bool(args.conserved_Sz), bool(args.warm_up), args.Q_tar
    conserved_C4 = bool(args.conserved_C4)
    conserved_inv = bool(args.conserved_inv)
    conserved_SU2, chem_pot = bool(args.conserved_SU2), args.chem_pot
    if not conserved_Sz:
        assert Q_tar is None
    else:
        assert Q_tar is not None


    if opt == "KFAC":
        KFAC = True
    else:
        KFAC = False

    if dim == 1:
        systemSize = (L, 2)
    elif dim == 2:
        systemSize = (L, L, 2)
        if H == 'Julian':
            systemSize = (L, L, 3)
        else:
            pass

    else:
        raise NotImplementedError

    Net = tf_network(which_net, systemSize, optimizer=opt, dim=dim, alpha=alpha,
                     activation=act, using_complex=using_complex, single_precision=SP,
                     batch_size=num_sample, num_blocks=num_blocks, multi_gpus=multi_gpus,
                     conserved_C4=conserved_C4, conserved_Sz=conserved_Sz, Q_tar=Q_tar,
                     conserved_SU2=conserved_SU2, chem_pot=chem_pot,
                     conserved_inv=conserved_inv,
                    )

    # args = parse_args()
    # L = args.L
    # which_net = args.which_net
    # lr = args.lr
    # batch_size = args.batch_size
    # J2 = args.J2
    # alpha = args.alpha
    # reg = args.reg
    # act = args.act
    # using_complex = args.using_complex
    # opt = args.opt  # "Mom"
    # conserved_Sz, warm_up, Q_tar = bool(args.conserved_Sz), bool(args.warm_up), args.Q_tar


    # system_size = (L, L, 2)
    # Net = tf_network(which_net, system_size, optimizer=opt,
    #                  dim=2, alpha=alpha, activation=act,
    #                  using_complex=using_complex,
    #                  Q_tar=Q_tar
    #                 )

    X_computation_basis = np.genfromtxt('ExactDiag/basis_L%d.csv' % (L**2), delimiter=',')
    X = np.zeros([2**(L**2), L, L, 2])
    X[:,:,:,0] = X_computation_basis.reshape([2**(L**2), L, L])
    X[:,:,:,1] = 1-X_computation_basis.reshape([2**(L**2), L, L])
    Y = np.genfromtxt('ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')
    Y = np.array(Y, dtype=np.complex128)[:, None]
    pi = tf.constant(np.pi)

    print(X.shape, Y.shape)

    with Net.sess as sess:

        true_amp_log_re = tf.placeholder(tf.float32, [None, 1])
        true_amp_log_im = tf.placeholder(tf.float32, [None, 1])
        net_amp_log_re = tf.real(Net.log_amp)
        net_amp_log_im = tf.imag(Net.log_amp)
        assert net_amp_log_re.shape[1] == 1
        ###################################
        # Cost function : KL + L2 #
        ###################################
        kl_cost = tf.reduce_mean(2 * (true_amp_log_re - net_amp_log_re))
        # l2_cost = tf.reduce_mean(tf.square( tf.mod(((true_amp_log_im - net_amp_log_im) + pi), 2*pi) - pi ))
        # l2_cost = tf.reduce_mean(tf.square( (true_amp_log_im - tf.mod(net_amp_log_im, 2*pi) )))
        l2_cost = tf.reduce_mean(tf.square(tf.cos(true_amp_log_im) - tf.cos(net_amp_log_im)) +
                                 tf.square(tf.sin(true_amp_log_im) - tf.sin(net_amp_log_im)))
        cost = kl_cost + l2_cost


        true_amp = tf.placeholder(tf.complex64, [None, 1])
        v1 = true_amp
        v2 = Net.amp
        # sampled_l2 = tf.reduce_mean(tf.square(tf.abs(v1 - v2)/tf.abs(v1)))
        sampled_fidelity = tf.reduce_mean(tf.real( v2/v1 ))
        # cost = -sampled_fidelity

        ###################################
        # Cost function 1: Batch fidelity #
        ###################################
        # cost = -tf.reduce_sum(tf.multiply(v1, v2))/tf.norm(v1)/tf.norm(v2)

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
        # ckpt_path = 'wavefunction/vmc2d/'+which_net+'_'+act+'/L'+str(L)+'a'+str(alpha)
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

        ##############################################################################
        import sys
        sys.path.append('/tuph/t30/space/ga63zuh/qTEBD/sampling_mps/')
        sys.path.append('/tuph/t30/space/ga63zuh/qTEBD/')
        import mps_func
        import qTEBD
        import sample_mps

        state = np.genfromtxt('/home/t30/all/ga63zuh/Neural-Network-Quantum-State/ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')
        MPS = mps_func.state_2_MPS(state, 16, 1000)

        MPS, err = qTEBD.right_canonicalize(MPS, no_trunc=True, chi=1000, normalized=True)
        print("err = ", err)

        #PIJ
        sample_mps.ipj_trans_pij(MPS)
        #IPJ

        #################################################################################

        for i in range(100000+1):
            # batch_mask = np.random.choice(len(Y), batch_size, p=p)
            # X_mini_batch = X[batch_mask]
            # Y_mini_batch = Y[batch_mask]

            configs, amps = sample_mps.parallel_sampling(MPS)
            Y_mini_batch = np.array(amps, dtype=np.complex128)[:, None]
            X_mini_batch = np.zeros([Y_mini_batch.size, L, L, 2])
            X_mini_batch[:, :, :, 0] = configs.reshape([-1, L, L])
            X_mini_batch[:, :, :, 1] = 1 - configs.reshape([-1, L, L])




            if i % 5000 == 0:
                ### get full batch information
                y = Net.get_amp(X)
                print(('y norm : ', np.linalg.norm(y)))
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
                    plt.plot(Y[mask]/np.linalg.norm(Y[mask]), '-o')
                    plt.plot(y[mask].real/np.linalg.norm(y[mask].real), '--')
                    plt.show()
                pass

            _, c, c1, c2, f = sess.run([train_step, cost, kl_cost, l2_cost, sampled_fidelity],
                                       feed_dict={Net.x: X_mini_batch,
                                                  true_amp_log_re: np.real(np.log(Y_mini_batch)),
                                                  true_amp_log_im: np.imag(np.log(Y_mini_batch)),
                                                  true_amp: Y_mini_batch})
            batch_cos_accu.append(-c)
            if i % 50 == 0:
                print("iter=", i, "cost=", c, "kl_cost=", c1, "l2_cost=", c2, "F=", f)

            if i % 500 == 0:
                print(("step:", i, " cosine accuracy:", -c, "Y norm",
                       np.linalg.norm(Y_mini_batch), "y norm:", np.linalg.norm(y)))
                # "dot:",
                # y.T.dot(Y_mini_batch)[0]/np.linalg.norm(Y_mini_batch)/np.linalg.norm(y))
                saver.save(sess, ckpt_path + '/pre')

    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_batch.csv' % (L, which_net, alpha, opt, lr),
               batch_cos_accu, '%.4e', delimiter=',')
    np.savetxt('log/pretrain/L%d_%s_a%s_%s%.e_total.csv' % (L, which_net, alpha, opt, lr),
               total_cos_accu, '%.4e', delimiter=',')

    # fig.savefig('L16_pretrain.eps',bbox_inches='tight')
