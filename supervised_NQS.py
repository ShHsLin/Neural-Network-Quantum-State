import os
import tensorflow as tf
import numpy as np

from utils.parse_args import parse_args
from network.tf_network import tf_network
import network.tf_wrapper as tf_
import VMC
import sys
sys.path.append('ExactDiag')
import many_body
import time


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

    assert (args.alpha is not None) ^ (args.alpha_list is not None)  # XOR
    if args.alpha is not None:
        alpha = args.alpha
        alpha_list = None
    else:
        alpha = None
        alpha_list = args.alpha_list

    filter_size = args.filter_size
    opt, batch_size, H, dim, num_iter = (args.opt, args.batch_size,
                                         args.H, args.dim, args.num_iter)
    PBC = args.PBC
    supervised_model = args.supervised_model
    assert supervised_model is not None

    num_blocks, multi_gpus = args.num_blocks, args.multi_gpus
    num_threads = args.num_threads
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
                     alpha_list=alpha_list, filter_size=filter_size,
                     activation=act, using_complex=using_complex, single_precision=SP,
                     batch_size=batch_size, num_blocks=num_blocks, multi_gpus=multi_gpus,
                     conserved_C4=conserved_C4, conserved_Sz=conserved_Sz, Q_tar=Q_tar,
                     conserved_SU2=conserved_SU2, chem_pot=chem_pot,
                     conserved_inv=conserved_inv, num_threads=num_threads
                     )

    if dim == 1:
        vmc = VMC.VMC_1d(systemSize, Wavefunction=Net, Hamiltonian=H, batch_size=batch_size,
                         J2=J2, reg=reg, using_complex=using_complex, single_precision=SP,
                         real_time=real_time, pinv_rcond=pinv_rcond, PBC=PBC)
    elif dim == 2:
        vmc = VMC.VMC_2d(systemSize, Net=Net, Hamiltonian=H, batch_size=batch_size,
                         J2=J2, reg=reg, using_complex=using_complex, single_precision=SP,
                         real_time=real_time, pinv_rcond=pinv_rcond, PBC=PBC)
    else:
        print("DIM error")
        raise NotImplementedError

    #####################################
    ## From ED;  Creating exact basis  ##
    #####################################
    ED = True
    if ED:
        if dim == 1:
            N_sys = L
            X_computation_basis = np.genfromtxt('ExactDiag/basis_L%d.csv' % L, delimiter=',')
            X = np.zeros([2**L, L, 2])
            X[:, :, 0] = X_computation_basis.reshape([2**L, L])
            X[:, :, 1] = 1-X_computation_basis.reshape([2**L, L])
            # Y = np.genfromtxt('ExactDiag/EigVec/ES_L'+str(L)+'_J2_'+str(int(J2*10))+'_OBC.csv').reshape((2**L, 1))
            # Y = np.genfromtxt('ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')

            Y = np.load('ExactDiag/wavefunction/%s/ED_wf_T%.2f.npy' % (supervised_model, args.T))
            Y = np.array(Y, dtype=np.complex128)[:, None]
        elif dim == 2:
            N_sys = L ** 2
            X_computation_basis = np.genfromtxt('ExactDiag/basis_L%d.csv' % (L**2), delimiter=',')
            X = np.zeros([2**(L**2), L, L, 2])
            X[:, :, :, 0] = X_computation_basis.reshape([2**(L**2), L, L])
            X[:, :, :, 1] = 1-X_computation_basis.reshape([2**(L**2), L, L])

            # Y = np.genfromtxt('ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')
            Y = np.load('ExactDiag/wavefunction/%s/ED_wf_T%.2f.npy' % (supervised_model, args.T))
            Y = np.array(Y, dtype=np.complex128)[:, None]
        else:
            raise NotImplementedError

        print(X.shape, Y.shape)
        ED_prob = (np.abs(Y)**2).flatten()
        # p = p + 100./batch_size
        # p = p/sum(p)
    else:
        ##############################################################################
        ## MPS Sampling; do not have the full Y ; only estimate by sampled overlap ##
        ##############################################################################
        import sys
        sys.path.append('/tuph/t30/space/ga63zuh/qTEBD/sampling_mps/')
        sys.path.append('/tuph/t30/space/ga63zuh/qTEBD/')
        import mps_func
        import qTEBD
        import sample_mps

        # Example 1
        # state = np.genfromtxt('/home/t30/all/ga63zuh/Neural-Network-Quantum-State/ExactDiag/EigVec/ES_2d_L4x4_J2_0.csv', delimiter=',')
        # MPS = mps_func.state_2_MPS(state, 16, 1000)
        # MPS, err = qTEBD.right_canonicalize(MPS, no_trunc=True, chi=1000, normalized=True)
        # print("err = ", err)

        # Example 2:
        import pickle
        MPS = np.load('ExactDiag/wavefunction/%s/somethinghere' % args.T)
        # MPS = pickle.load(open('/tuph/t30/space/ga63zuh/qTEBD/tenpy_tebd/data_tebd_dt1.000000e-03/1d_TFI_g1.4000_h0.9045/L31/wf_chi1024_4th/T%.1f.pkl' % args.T, 'rb'))
        # MPS = [np.array([1., 0.]).reshape([2, 1, 1])] * 31

        # PIJ
        sample_mps.ipj_trans_pij(MPS)
        # IPJ
        #################################################################################

    t0 = time.time()
    with Net.sess as sess:
        pi = tf.constant(np.pi, dtype=Net.TF_FLOAT)

        true_amp_log_re = tf.placeholder(Net.TF_FLOAT, [None, 1])
        true_amp_log_im = tf.placeholder(Net.TF_FLOAT, [None, 1])
        net_amp_log_re = tf.real(Net.log_amp)
        net_amp_log_im = tf.imag(Net.log_amp)
        assert net_amp_log_re.shape[1] == 1
        ###################################
        # Cost function : KL + L2 #
        ###################################
        kl_cost = tf.reduce_mean(2 * (true_amp_log_re - net_amp_log_re))
        l2_cost_old = tf.reduce_mean(tf.abs(tf.mod(((true_amp_log_im - net_amp_log_im) + pi), 2*pi) - pi))
        # l2_cost = tf.reduce_mean(tf.square( (true_amp_log_im - tf.mod(net_amp_log_im, 2*pi) )))
        l2_cost = tf.reduce_mean(tf.square(tf.cos(true_amp_log_im) - tf.cos(net_amp_log_im)) +
                                 tf.square(tf.sin(true_amp_log_im) - tf.sin(net_amp_log_im)))
        cost = kl_cost + l2_cost

        true_amp = tf.placeholder(Net.TF_COMPLEX, [None, 1])
        v1 = true_amp
        v2 = Net.amp
        # sampled_l2 = tf.reduce_mean(tf.square(tf.abs(v1 - v2)/tf.abs(v1)))
        normalized_target_sampled_overlap = tf.reduce_mean(tf.real(v2/v1))
        normalized_target_sampled_fidelity = tf.square(tf.abs(tf.reduce_mean(v2/v1)))
        uniform_sampled_fidelity = tf.square(tf.abs(tf.reduce_mean(tf.conj(v2)*v1))) / tf.real(tf.reduce_mean(tf.conj(v2)*v2)) / tf.real(tf.reduce_mean(tf.conj(v1)*v1))
        target_sampled_fidelity = tf.real(tf.reduce_mean(v2/v1) * tf.reduce_mean(tf.conj(v2)/tf.conj(v1)) / tf.reduce_mean( (tf.conj(v2)*v2) / (tf.conj(v1)*v1) ))

        if args.cost_function == 'joint':
            assert args.sampling_dist == 'target'
            # This works for sampling according to the target wf given.
            sampled_fidelity = normalized_target_sampled_fidelity
            cost = kl_cost + l2_cost
        elif args.cost_function == 'neg_F' and args.sampling_dist == 'target':
            # This works for sampling according to the target wf given.
            sampled_fidelity = target_sampled_fidelity
            cost = - sampled_fidelity
        elif args.cost_function == 'neg_F' and args.sampling_dist == 'uniform':
            # This works for full batch or uniform sampling.
            sampled_fidelity = uniform_sampled_fidelity
            cost = - sampled_fidelity
        else:
            raise NotImplementedError



        # cost = 1. - normalized_target_sampled_overlap
        # cost = 1. - sampled_fidelity
        true_log_amp = tf.complex(true_amp_log_re, true_amp_log_im)
        # cost = tf.reduce_sum(tf.square( tf.abs( true_log_amp - Net.log_amp )))
        # cost = tf.reduce_sum(tf.square( tf.abs( tf.exp(true_log_amp) - tf.exp(Net.log_amp))))
        # cost = tf.reduce_mean(tf.square( tf.abs( tf.exp(true_log_amp) - tf.exp(Net.log_amp)) /  tf.square(tf.abs(tf.exp(true_log_amp))) ))
        # cost = - tf.log(sampled_fidelity)

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
        if alpha is not None:
            ckpt_path = path + \
                'wavefunction/Supervised/' + '%s_T%.2f/' % (supervised_model, args.T) + \
                which_net+'_'+act+'_L'+str(L)+'_a'+str(alpha)
            # 'wavefunction/Supervised/' + 'TE_TFI_h0.9045_T%.1f/' % args.T + ...
        else:
            ckpt_path = path + \
                'wavefunction/Supervised/' + '%s_T%.2f/' % (supervised_model, args.T) + \
                which_net+'_'+act+'_L'+str(L)+'_a'+('-'.join([str(alpha) for alpha in alpha_list]))

        if num_blocks is not None:
            ckpt_path = ckpt_path + '_block%d' % num_blocks

        if filter_size is not None:
            ckpt_path = ckpt_path + '_f%d' % filter_size

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

        try:
            data_dict = np.load(ckpt_path + '/data_dict.npy', allow_pickle=True).item()
            print("found data_dict")
        except:
            print("no data_dict found; create new data_dict")
            data_dict = {'cost_avg': [], 'kl_avg': [], 'l2_avg': [], 'fidelity_avg': [],
                         'cost_var': [], 'kl_var': [], 'l2_var': [], 'fidelity_var': [],
                         'lr': [], 'timestamps': [],
                         }
            data_dict['num_para'] = Net.getNumPara()

        cost_list = []
        kl_list = []
        l2_list = []
        fidelity_list = []
        for i in range(1, num_iter+1):
            if ED and (not args.exact_gradient):
                if args.sampling_dist == 'target':
                    ## sampling based on target distribution
                    batch_mask = np.random.choice(len(Y), batch_size, p=ED_prob)
                elif args.sampling_dist == 'uniform':
                    ## unifrom sampling
                    batch_mask = np.random.choice(len(Y), batch_size)
                else:
                    raise NotImplementedError

                X_mini_batch = X[batch_mask]
                Y_mini_batch = Y[batch_mask]
            elif ED and args.exact_gradient:
                ## FULL BATCH HERE
                X_mini_batch = X
                Y_mini_batch = Y
            else:
                configs, amps = sample_mps.parallel_sampling(MPS, batch_size)
                Y_mini_batch = np.array(amps, dtype=np.complex128)[:, None]
                X_mini_batch = np.zeros([batch_size, *systemSize])
                X_mini_batch[..., 0] = configs.reshape([batch_size, *systemSize[:-1]])
                X_mini_batch[..., 1] = 1 - configs.reshape([batch_size, *systemSize[:-1]])

            _, c, kl, l2, f = sess.run([train_step, cost, kl_cost, l2_cost, sampled_fidelity],
                                       feed_dict={Net.x: X_mini_batch,
                                                  true_amp_log_re: np.real(np.log(Y_mini_batch)),
                                                  true_amp_log_im: np.imag(np.log(Y_mini_batch)),
                                                  true_amp: Y_mini_batch})

            cost_list.append(c)
            kl_list.append(kl)
            l2_list.append(l2)
            fidelity_list.append(f)
            if i % 500 == 0:
                data_dict['lr'].append(lr)
                data_dict['cost_avg'].append(np.average(cost_list))
                data_dict['kl_avg'].append(np.average(kl_list))
                data_dict['l2_avg'].append(np.average(l2_list))
                data_dict['fidelity_avg'].append(np.average(fidelity_list))
                data_dict['cost_var'].append(np.var(cost_list))
                data_dict['kl_var'].append(np.var(kl_list))
                data_dict['l2_var'].append(np.var(l2_list))
                data_dict['fidelity_var'].append(np.var(fidelity_list))
                data_dict['timestamps'].append(time.time() - t0)
                print("iter=", i, "cost=", data_dict['cost_avg'][-1],
                      "kl_cost=", data_dict['kl_avg'][-1],
                      "l2_cost=", data_dict['l2_avg'][-1],
                      "F = ", data_dict['fidelity_avg'][-1])
                cost_list = []
                kl_list = []
                l2_list = []
                fidelity_list = []

            if ED and i % 5000 == 0:
                # get full batch information
                # y = Net.get_amp(X)
                y_list = []
                end_idx = 0
                for i in range((2**N_sys) // 1024):
                    start_idx, end_idx = i*1024, (i+1)*1024
                    yi = Net.get_amp(X[start_idx:end_idx])
                    y_list.append(yi)

                if end_idx != 2**N_sys:
                    y_list.append(Net.get_amp(X[end_idx:]))

                y = np.concatenate(y_list)

                y_norm = np.linalg.norm(y)
                print('y norm : ', y_norm)
                measured_fidelity = np.square(np.abs(Y.flatten().dot(y.flatten().conj())) / y_norm )
                print("Fidelity = ", measured_fidelity)
                data_dict['fidelity'] = measured_fidelity

                saver.save(sess, ckpt_path + '/pre')
                np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

                # if measured_fidelity > 1 - 1e-4 or (measured_fidelity - previous_fidelity) < 1e-4 :
                #     break
                # else:
                #     previous_fidelity = measured_fidelity
                if (measured_fidelity > 1 - 1e-4) or (data_dict['cost_avg'][-1] > np.average(data_dict['cost_avg'][-10:-5])):
                    break

                PLOT = False
                SZ_MASK = True
                if dim == 1:
                    mask = np.sum(X[:, :, 0], axis=(1)) == 8
                elif dim == 2:
                    mask = np.sum(X[:, :, :, 0], axis=(1, 2)) == 8
                y_mask = y[mask]
                print("Sz 0 prob : ", y_mask.T.conjugate().dot(y_mask) / (y_norm**2))
                if PLOT and SZ_MASK:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(Y[mask]/np.linalg.norm(Y[mask]), '-o')
                    plt.plot(y[mask].real/np.linalg.norm(y[mask].real), '--')
                    plt.show()
                    import pdb
                    pdb.set_trace()
                elif PLOT:
                    import matplotlib.pyplot as plt
                    fig = plt.figure()
                    plt.plot(Y/np.linalg.norm(Y), '-o')
                    plt.plot(y/np.linalg.norm(y), '--')
                    plt.show()

        if ED:
            sx_expectation = many_body.sx_expectation(N_sys//2+1, y.flatten(), N_sys)
            print("<Sx> = ", sx_expectation)
            data_dict["Sx"] = sx_expectation

            S2, SvN = many_body.entanglement_entropy(N_sys//2 + 1, y.flatten(), N_sys)
            data_dict["renyi_2"] = S2
            data_dict["SvN"] = SvN
        else:
            ####################################################
            ### Measure renyi-2 entanglement entropy  ###
            ####################################################
            config_arr = np.zeros([2 * num_sample, *systemSize])
            for i in range(1, 1+int(2 * num_sample // batch_size)):
                vmc.forward_sampling()
                config_arr[(i-1) * batch_size: i * batch_size] = vmc.config.copy()

            amp_no_swap = Net.get_amp(config_arr)

            swap_config_arr = config_arr.copy()
            swap_config_arr[:num_sample, :L//2, :] = config_arr[num_sample:, :L//2, :]
            swap_config_arr[num_sample:, :L//2, :] = config_arr[:num_sample, :L//2, :]
            amp_swap = Net.get_amp(swap_config_arr)

            rho_2 = 0
            for i in range(num_sample):
                rho_2 += (amp_swap[i].conj() * amp_swap[i+num_sample].conj()) / \
                    (amp_no_swap[i].conj() * amp_no_swap[i+num_sample].conj())

            rho_2 = rho_2[0] / num_sample
            print("Tr rho_2 = ", rho_2)
            print("Renyi 2 = ", -np.log(rho_2))

            print(type(rho_2))
            # data_dict = np.load(ckpt_path+'/data_dict.npy', allow_pickle=True).item()
            data_dict["renyi_2"] = -np.log(rho_2)

            #################
            # Measure Sx
            #################

            flip_config_arr = config_arr.copy()
            flip_config_arr[:, L//2, :] = (1 - flip_config_arr[:, L//2, :])

            amp_flip = Net.get_amp(flip_config_arr)
            # amp_no_swap
            sx_expectation = np.average(amp_flip / amp_no_swap)
            print("<Sx> = ", sx_expectation)
            data_dict["Sx"] = sx_expectation

        saver.save(sess, ckpt_path + '/pre')
        np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)
