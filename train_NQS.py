# from memory_profiler import profile
import os
import sys
import pickle
import time
import numpy as np
import tensorflow as tf
from utils.parse_args import parse_args
from network.tf_network import tf_network
import VMC

from tensorflow.python.training.saver import BaseSaverBuilder


class CastFromFloat32SaverBuilder(BaseSaverBuilder):
    # Based on tensorflow.python.training.saver.BulkSaverBuilder.bulk_restore
    def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                     restore_sequentially):
        from tensorflow.python.ops import io_ops
        restore_specs = []
        for saveable in saveables:
            for spec in saveable.specs:
                restore_specs.append((spec.name, spec.slice_spec, spec.dtype))
        names, slices, dtypes = zip(*restore_specs)
        restore_dtypes = [tf.float32 for _ in dtypes]
        # with tf.device("cpu:0"):
        restored = io_ops.restore_v2(filename_tensor, names, slices, restore_dtypes)
        return [tf.cast(r, dt) for r, dt in zip(restored, dtypes)]


def save_result_dict(result_filename, tmp_result_dict, info_dict, iteridx, save_each):
    if not os.path.isfile(result_filename):
        result_dict = {}
        for key in info_dict.keys():
            if key not in ["num_p_unique", "num_p_counts"]:
                result_dict[key] = np.empty([0, *tmp_result_dict[key][0].shape])
            else:
                result_dict[key] = []

    else:
        result_dict = pickle.load(open(result_filename, 'rb'))

    for key in info_dict.keys():
        if key not in ["num_p_unique", "num_p_counts"]:
            assert result_dict[key].shape[0] == iteridx - save_each
            result_dict[key] = np.concatenate([result_dict[key],
                                               tmp_result_dict[key][-save_each:]]
                                              )
        else:
            result_dict[key].append(info_dict[key])

    pickle.dump(result_dict, open(result_filename, 'wb'))
    return


def progress(count, total, info_dict=None, head=False):
    if head:
        print("-"*16 + " bar " + "-"*11 + " percent ,      < E >       ,     std < E > ,   | G |,  max_amp")
    else:
        bar_len = 30
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '>' * filled_len + '-' * (bar_len - filled_len)

        Eavg = info_dict['E0']
        Evar = info_dict['E0_var']
        max_amp = info_dict['max_amp']
        G_norm = info_dict['G_norm']

        if 'totalS' in info_dict.keys():
            totalS = info_dict['totalS']
            sys.stdout.write('\r[%s] %s%s , %g+%gj, %g , %g , %g+%gj, %g+%gj' % (bar, percents, '%', Eavg.real,
                                                                                 Eavg.imag, np.sqrt(Evar), G_norm,
                                                                                 max_amp.real, max_amp.imag, totalS.real, totalS.imag))
        else:
            sys.stdout.write('\r[%s] %s%s , %g+%gj, %g , %g , %g+%gj' % (bar, percents, '%', Eavg.real,
                                                                         Eavg.imag, np.sqrt(Evar), G_norm,
                                                                         max_amp.real, max_amp.imag))

        sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)


def dw_to_glist(GradW, var_shape_list):
    grad_list = []
    grad_ind = 0
    for var_shape in var_shape_list:
        var_size = np.prod(var_shape)
        grad_list.append(
            GradW[grad_ind:grad_ind + var_size].reshape(var_shape))
        grad_ind += var_size

    return grad_list


if __name__ == "__main__":
    ###############################
    #  Read the input argument ####
    ###############################
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
                     batch_size=num_sample, num_blocks=num_blocks, multi_gpus=multi_gpus,
                     conserved_C4=conserved_C4, conserved_Sz=conserved_Sz, Q_tar=Q_tar,
                     conserved_SU2=conserved_SU2, chem_pot=chem_pot,
                     conserved_inv=conserved_inv, num_threads=num_threads,
                     )
    if dim == 1:
        vmc = VMC.VMC_1d(systemSize, Wavefunction=Net, Hamiltonian=H, batch_size=batch_size,
                         J2=J2, reg=reg, using_complex=using_complex, single_precision=SP,
                         real_time=real_time, pinv_rcond=pinv_rcond, PBC=PBC)
    elif dim == 2:
        vmc = VMC.VMC_2d(systemSize, Wavefunction=Net, Hamiltonian=H, batch_size=batch_size,
                         J2=J2, reg=reg, using_complex=using_complex, single_precision=SP,
                         real_time=real_time, pinv_rcond=pinv_rcond, PBC=PBC)
    else:
        print("DIM error")
        raise NotImplementedError

    # Run Initilizer
    vmc.wf.run_global_variables_initializer()

    print("Total num para: ", vmc.wf_num_para)
    if SR:
        print("Using Stochastic Reconfiguration")
        if vmc.wf_num_para / 1 < num_sample:
            print("forming Sij explicitly")
            explicit_SR = True
        else:
            print("DO NOT FORM Sij explicity")
            explicit_SR = False
    else:
        explicit_SR = None
        print("Using plain gradient descent")

    var_shape_list = vmc.wf.var_shape_list
    var_list = tf.global_variables()

    #############################################################
    # OLD CHECK_POINT FORMAT
    # ckpt_path = path + 'wavefunction/Pretrain/%s/L%d/' % (which_net, L)
    # ckpt_path = path + \
    #     'wavefunction/vmc%dd/%s_%s/L%da%d/' % (dim,
    #                                            which_net, act, L, alpha)
    if alpha is not None:
        ckpt_path = path + \
            'wavefunction/vmc%dd_%s_L%d/%s_%s_a%d' % (dim, H, L, which_net, act, alpha)
    else:
        ckpt_path = path + \
            'wavefunction/vmc%dd_%s_L%d/%s_%s_a' % (dim, H, L, which_net, act) + \
            ('-'.join([str(alpha) for alpha in alpha_list]))

    if filter_size is not None:
        ckpt_path = ckpt_path + '_f%d/' % filter_size
    else:
        ckpt_path = ckpt_path + '/'

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    #############################################################

    try:
        # saver = tf.train.Saver(vmc.wf.model_var_list)
        saver = tf.train.Saver()

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(vmc.wf.sess, ckpt.model_checkpoint_path)
            print("Restore from last check point, stored at %s" % ckpt_path)
            # print(vmc.wf.sess.run(vmc.wf.para_list))
        else:
            print("No checkpoint found, at %s " % ckpt_path)
    except Exception as e:
        print(e)
        print("import weights only, not include stabilier, may cause numerical instability")
        saver = tf.train.Saver(vmc.wf.para_list, builder=CastFromFloat32SaverBuilder())

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(vmc.wf.sess, ckpt.model_checkpoint_path)
            print("Restore from last check point, stored at %s" % ckpt_path)
            # print(vmc.wf.sess.run(vmc.wf.para_list))
        else:
            print("No checkpoint found, at %s " % ckpt_path)

        # saver = tf.train.Saver(vmc.wf.model_var_list)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(ckpt_path)

        diag_QFI_path = path + 'L%d_%s_%s_a%s_%s%.e_S%d_diag_QFI.npy' % (L, which_net, act, alpha, opt, lr, num_sample)
        if os.path.isfile(diag_QFI_path):
            vmc.diag_QFI = np.load(path + 'L%d_%s_%s_a%s_%s%.e_S%d_diag_QFI.npy' %
                                   (L, which_net, act, alpha, opt, lr, num_sample),
                                   vmc.diag_QFI)

    # Thermalization
    print("Thermalizing ~~ ")
    start_t, start_c = time.time(), time.clock()
    # vmc.update_stabilizer()
    if 'pixelCNN' not in which_net:
        if batch_size > 1:
            for i in range(1000):
                vmc.new_config_batch()
        else:
            for i in range(1000):
                vmc.new_config()

    end_t, end_c = time.time(), time.clock()
    print("Thermalization time: ", end_c - start_c, end_t - start_t)

    tmp_result_dict = {"E": [], "E_var": [], "E0": [], "E0_var": [],
                       "G_norm": [],
                       "max_amp": [], "channel_stat": [],
                       }
    if conserved_SU2:
        tmp_result_dict["totalS"] = []
        tmp_result_dict["totalS_var"] = []

    vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr))
    vmc.wf.sess.run(vmc.wf.momentum.assign(0.9))
    GradW = None
    # vmc.moving_E_avg = E_avg * l

    warm_up_array = np.ones(num_iter)  # np.ones(num_iter)
    if warm_up:
        assert num_iter > 2000
        warm_up_array[:2000] = np.arange(0.1, 1, 0.9/2000)

    progress(0, num_iter, None, head=True)

    result_filename = path + 'L%d_%s_%s_a%s_%s%.e_S%d_noSR.pkl' % (L, which_net, act, alpha, opt, lr, num_sample)
    if not os.path.isfile(result_filename):
        start_idx = 0
    else:
        result_dict = pickle.load(open(result_filename, 'rb'))
        start_idx = result_dict['E'].shape[0]

    for iteridx in range(start_idx + 1, num_iter + 1):
        if warm_up:
            vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr*warm_up_array[iteridx-1]))

        # vmc.update_stabilizer()

        # vmc.wf.sess.run(vmc.wf.weights['wc1'].assign(wc1))
        # vmc.wf.sess.run(vmc.wf.biases['bc1'].assign(bc1))

        #    vmc.wf.sess.run(vmc.wf.learning_rate.assign(1e-3 * (0.995**iteridx)))
        #    vmc.wf.sess.run(vmc.wf.momentum.assign(0.95 - 0.4 * (0.98**iteridx)))
        # num_sample = 500 + iteridx/10

        GradW, GjFj, info_dict = vmc.get_VMC_gradient(num_sample=num_sample, iteridx=iteridx,
                                                      SR=SR, Gj=GradW, explicit_SR=explicit_SR,
                                                      KFAC=KFAC)

        progress(iteridx, num_iter, info_dict)

        if not real_time:
            if GjFj is not None:
                # Trust region method:
                # GradW = GradW / np.sqrt(iteridx)
                if lr > lr / np.sqrt(GjFj):
                    GradW = GradW / np.sqrt(GjFj)
                # TR_lr = np.amin([1e-1, 1e-3/np.sqrt(GjFj)])
                # GradW = GradW * TR_lr / lr
            else:
                pass
                # if np.linalg.norm(GradW) > 100:
                #     GradW = GradW / np.linalg.norm(GradW) * 100

            # GradW = GradW/np.linalg.norm(GradW)*np.amax([(0.95**iteridx),0.1])

            # GradW = np.random.rand(*GradW.shape) * np.sign(GradW) * np.amax([(0.95**iteridx),1e-2])
            # GradW = GradW/np.linalg.norm(GradW)*np.amax([(0.97**iteridx),1e-3])
            # if np.linalg.norm(GradW) > 1000:
            #    GradW = GradW/np.linalg.norm(GradW) * 1000
        else:  # real-time
            if integration == 'mid_point':
                mid_pt_iter = 0
                conv = False
                while(mid_pt_iter < 20 and not conv):
                    grad_list = dw_to_glist(GradW, var_shape_list)
                    vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr / 2.))
                    vmc.wf.applyGrad(grad_list)
                    GradW_mid, E, E_var, GjFj = vmc.get_VMC_gradient(num_sample=num_sample,
                                                                     iteridx=iteridx,
                                                                     SR=SR, Gj=GradW,
                                                                     explicit_SR=explicit_SR)
                    if np.linalg.norm(GradW - GradW_mid) / np.linalg.norm(GradW) < 1e-6:
                        conv = True
                    else:
                        print("iter=", mid_pt_iter, " not conv yet : err = ",
                              np.linalg.norm(GradW - GradW_mid) / np.linalg.norm(GradW))

                    grad_list = dw_to_glist(-GradW, var_shape_list)
                    vmc.wf.applyGrad(grad_list)
                    vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr))
                    GradW = GradW_mid
                    mid_pt_iter += 1

            elif integration == 'rk4':
                # x0
                k1 = GradW
                grad_list = dw_to_glist(GradW, var_shape_list)
                # Step size = h/2
                vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr / 2.))
                vmc.wf.applyGrad(grad_list)
                # x0 + k1 * h/2
                GradW_2, E, E_var, GjFj = vmc.get_VMC_gradient(num_sample=num_sample,
                                                               iteridx=iteridx,
                                                               SR=SR, Gj=GradW,
                                                               explicit_SR=explicit_SR)
                k2 = GradW_2
                grad_list = dw_to_glist(-GradW + GradW_2, var_shape_list)
                vmc.wf.applyGrad(grad_list)
                # x0 + k2 * h/2
                GradW_3, E, E_var, GjFj = vmc.get_VMC_gradient(num_sample=num_sample,
                                                               iteridx=iteridx,
                                                               SR=SR, Gj=GradW,
                                                               explicit_SR=explicit_SR)
                k3 = GradW_3
                grad_list = dw_to_glist(-GradW_2, var_shape_list)
                vmc.wf.applyGrad(grad_list)
                # x0
                # Step size = h
                vmc.wf.sess.run(vmc.wf.learning_rate.assign(lr))
                grad_list = dw_to_glist(GradW_3, var_shape_list)
                vmc.wf.applyGrad(grad_list)
                # x0 + k3 * h
                GradW_4, E, E_var, GjFj = vmc.get_VMC_gradient(num_sample=num_sample,
                                                               iteridx=iteridx,
                                                               SR=SR, Gj=GradW,
                                                               explicit_SR=explicit_SR)
                k4 = GradW_4
                grad_list = dw_to_glist(-GradW_3, var_shape_list)
                vmc.wf.applyGrad(grad_list)
                # x0
                GradW = (k1 + 2 * k2 + 2 * k3 + k4) / 6.

            elif integration == 'explicit_euler':
                pass
            else:
                raise NotImplementedError

        # info_dict contain:
        #   "E0", "E0_var", "E", "E_var", "max_amp",
        #   "num_p_unique", "num_p_counts", "channel_stat"
        for key in info_dict.keys():
            if key not in ["num_p_unique", "num_p_counts"]:
                tmp_result_dict[key].append(info_dict[key])

        # [TODO] delete the code below
        # E_log.append(info_dict['E'])
        # E_var_log.append(info_dict['E_var'])
        # E0_log.append(info_dict['E0'])
        # E0_var_log.append(info_dict['E0_var'])
        # max_amp_log.append(info_dict['max_amp'])
        # channel_stat_log.append(info_dict['channel_stat'])
        # num_p_unique & num_p_counts only store per save_each steps.
        # to ease storage.

        # GradW = GradW/np.sqrt(iteridx)
        grad_list = dw_to_glist(GradW, var_shape_list)

        #  L2 Regularization ###
        # for idx, W in enumerate(vmc.wf.sess.run(vmc.wf.para_list)):
        #     grad_list[idx] += W * reg

        vmc.wf.applyGrad(grad_list)
        # To save object ##
        if iteridx % save_each == 0:
            # Saving WF
            if np.isnan(tmp_result_dict["E0"][-1]):
                print("nan in Energy, stop!")
                break
            else:
                print(" Wavefunction saved ~ ")
                saver.save(vmc.wf.sess, ckpt_path + 'opt%s_S%d' %
                           (opt, num_sample))
            # Saving E_list
            if SR:
                log_file = open(path + 'L%d_%s_%s_a%s_%s%.e_S%d.csv' %
                                (L, which_net, act, alpha, opt, lr, num_sample),
                                'a')
                np.savetxt(log_file, tmp_result_dict["E0"][-save_each:], '%.6e', delimiter=',')
                log_file.close()

                cov_s_list = []
                for cov in vmc.cov_list:
                    if type(cov) == list:
                        cov_s_list.append(cov[0])
                    else:
                        pass

                log_file = open(path + 'L%d_%s_%s_a%s_%s%.e_S%d_cov.csv' %
                                (L, which_net, act, alpha, opt, lr, num_sample),
                                'a')
                np.savetxt(log_file, np.concatenate(cov_s_list), '%.6e', delimiter=',')
                log_file.close()

            else:
                # [TODO] delete the code below
                # filename_csv = path + 'L%d_%s_%s_a%s_%s%.e_S%d_noSR.csv' % (L, which_net, act, alpha, opt, lr, num_sample)
                # log_file = open(filename_csv, 'a')
                # np.savetxt(log_file, tmp_result_dict["E0"][-save_each:], '%.6e', delimiter=',')
                # log_file.close()

                # result_filename = path + 'L%d_%s_%s_a%s_%s%.e_S%d_noSR.pkl' % (L, which_net, act, alpha, opt, lr, num_sample)

                save_result_dict(result_filename, tmp_result_dict, info_dict, iteridx, save_each)
        else:
            pass

    '''
    Task1
    Write down again the Probability assumption
    and the connection with deep learning model

    '''
