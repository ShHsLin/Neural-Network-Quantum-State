import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
import numpy as np
import tensorflow as tf
import time

"""
Should add spin-spin correlation in 2d
should add vmc_observable in 2d

1.  Should move config out as an indep class
So that easily to change from 1d problem to 2d problem?
2.  Rewrite the h, J,... etc in a class model
So easily to switch model
"""

class NQS_base():
    def __init__(self):
        self.cov_list = None
        return

    def list_to_vec(self, T_list):
        '''
        Transform a list of np.array to one single np.array (vector).
        '''
        return np.concatenate([t.flatten() for t in T_list])

    def vec_to_list(self, T_vec):
        '''
        assuming the vector is related to the weight.
        '''
        var_shape_list = self.NNet.var_shape_list
        T_list = []
        T_ind = 0
        for var_shape in var_shape_list:
            var_size = np.prod(var_shape)
            T_list.append(T_vec[T_ind:T_ind + var_size].reshape(var_shape))
            T_ind += var_size

        return T_list

    def Oarray_to_Olist(self, Oarray):
        '''
        We turn the Oarray of the shape (num_para, num_sample)
        to a list of shape
        [(size_para_1, num_sample), (size_para_2, num_sample), ...]
        '''
        num_para, num_sample = Oarray.shape
        var_shape_list = self.NNet.var_shape_list
        unaggregated_O_list = []
        T_ind = 0
        for var_shape in var_shape_list:
            var_size = np.prod(var_shape)
            unaggregated_O_list.append(Oarray[T_ind:T_ind + var_size, :].reshape([var_size, num_sample]))
            T_ind += var_size

        assert num_para == T_ind
        return unaggregated_O_list


    def getSelfAmp(self):
        return float(self.NNet.get_amp(self.config))

    def get_self_amp_batch(self):
        return self.NNet.get_amp(self.config).flatten()

    def get_self_log_amp_batch(self):
        return self.NNet.get_log_amp(self.config).flatten()

    def update_stabilizer(self):
        current_amp = self.get_self_amp_batch()
        max_abs_amp = np.max(np.abs(current_amp))
        log_max_abs_amp = np.log(max_abs_amp)
        print("exp_stabilier = %.5e, increments = %.5e " % (self.NNet.sess.run(self.NNet.exp_stabilizer),
                                                            log_max_abs_amp))
        self.NNet.exp_stabilizer_add(log_max_abs_amp)
        return

    def eval_amp_array(self, config_arr):
        '''
        Return the amplitude of the NNQS/NAQS with NNet function, get_amp.
        '''
        # for 1d:
        # (batch_size, inputShape[0], inputShape[1])
        # for 2d:
        # (batch_size, inputShape[0], inputShape[1], inputShape[2])
        array_shape = config_arr.shape
        max_size = self.max_batch_size
        if array_shape[0] <= max_size:
            return self.NNet.get_amp(config_arr).flatten()
        else:
            if self.using_complex:
                amp_array = np.empty((array_shape[0], ), dtype=self.NP_COMPLEX)
            else:
                amp_array = np.empty((array_shape[0], ), dtype=self.NP_FLOAT)

            for idx in range(array_shape[0] // max_size):
                amp_array[max_size * idx : max_size * (idx + 1)] = self.NNet.get_amp(config_arr[max_size * idx : max_size * (idx + 1)]).flatten()

            amp_array[max_size * (array_shape[0]//max_size) : ] = self.NNet.get_amp(config_arr[max_size * (array_shape[0]//max_size) : ]).flatten()
            return amp_array

    def eval_log_amp_array(self, config_arr):
        # for 1d:
        # (batch_size, inputShape[0], inputShape[1])
        # for 2d:
        # (batch_size, inputShape[0], inputShape[1], inputShape[2])
        array_shape = config_arr.shape
        max_size = self.max_batch_size
        if array_shape[0] <= max_size:
            return self.NNet.get_log_amp(config_arr).flatten()
        else:
            log_amp_array = np.empty((array_shape[0], ), dtype=np.complex64)
            for idx in range(array_shape[0] // max_size):
                log_amp_array[max_size * idx : max_size * (idx + 1)] = self.NNet.get_log_amp(config_arr[max_size * idx : max_size * (idx + 1)]).flatten()

            if array_shape[0] % max_size != 0:
                log_amp_array[max_size * (array_shape[0]//max_size) : ] = self.NNet.get_log_amp(config_arr[max_size * (array_shape[0]//max_size) : ]).flatten()

            return log_amp_array

    def VMC(self, num_sample, iteridx=0, SR=True, Gj=None, explicit_SR=False, KFAC=True,
            verbose=False,
           ):
        numPara = self.net_num_para
        num_site = self.num_site
        # OOsum = np.zeros((numPara, numPara))
        if self.using_complex:
            NP_DTYPE = self.NP_COMPLEX
        else:
            NP_DTYPE = self.NP_FLOAT

        Osum = np.zeros((numPara), dtype=NP_DTYPE)
        Earray = np.zeros((num_sample), dtype=NP_DTYPE)
        EOsum = np.zeros((numPara), dtype=NP_DTYPE)
        # Oarray = np.zeros((numPara, num_sample), dtype=NP_DTYPE)

        start_c, start_t = time.clock(), time.time()
        corrlength = self.corrlength
        configDim = list(self.config.shape)
        configDim[0] = num_sample
        config_arr = np.zeros(configDim, dtype=np.int8)

        NAQS = True
        if NAQS:
            for i in range(1, 1+int(num_sample//self.batch_size)):
                self.forward_sampling(sym_sec=None)
                config_arr[(i-1)*self.batch_size: i*self.batch_size] = self.config.copy()
        else:
            if (self.batch_size == 1):
                for i in range(1, 1 + num_sample * corrlength):
                    self.new_config()
                    if i % corrlength == 0:
                        config_arr[i // corrlength - 1] = self.config[0]

            else:
                sum_accept_ratio = 0
                for i in range(1, 1 + int(num_sample * corrlength / self.batch_size)):
                    ac = self.new_config_batch()
                    sum_accept_ratio += ac
                    bs = self.batch_size
                    if i % corrlength == 0:
                        i_c = i // corrlength
                        config_arr[(i_c-1)*bs: i_c*bs] = self.config[:]
                    else:
                        pass

                if verbose:
                    print("acceptance ratio: ", sum_accept_ratio/(1. + int(num_sample * corrlength / self.batch_size)))


        end_c, end_t = time.clock(), time.time()
        if verbose:
            print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)


        ### Statistics should be collected here
        ###
        ### 1. ) Log number of particle per sample.
        ### 2. ) Chemical potential, SU(2) constraint should be added here
        ###

        info_dict = {}
        ## config_arr [batch_size, Lx, Ly, channels] --> sum_config_arr [batch_size, channels]
        sum_config_arr = np.sum(config_arr, axis=(1,2))
        ## sum_to_channel [channels]  ; This gives statisics in channels sampled.
        sum_to_channel = np.sum(sum_config_arr, axis=0)
        ## num_particle [batch_size]
        num_particle = sum_config_arr.dot(np.arange(self.channels))
        num_p_unique, num_p_counts = np.unique(num_particle, return_counts=True)
        avg_num_particle = np.mean(num_particle)  # avg num per sample
        avg_num_particle_per_site = avg_num_particle / np.prod(configDim[1:-1])

        info_dict['num_p_unique'] = num_p_unique
        info_dict['num_p_counts'] = num_p_counts
        info_dict['channel_stat'] = sum_to_channel / np.prod(configDim[:-1])

        if verbose:
            print("sampled channel statistic : ", sum_to_channel / np.prod(configDim[:-1]))
            print("sampled config <n>/Lx/Ly : ", avg_num_particle_per_site)

        if self.NNet.conserved_Sz:
            assert np.isclose(np.sum(num_particle == self.NNet.Q_tar)/configDim[0], 1.)


        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(config_arr[i:i+1])
        E0array = self.get_local_E_batch(config_arr)
        # import pdb;pdb.set_trace()
        assert type(E0array) == np.ndarray

        Earray = E0array

        if self.NNet.conserved_SU2:
            totalS_array = self.local_totalS_batch_log(config_arr)

            totalS_avg = np.average(totalS_array)
            totalS_var = np.var(totalS_array)
            info_dict['totalS'] = totalS_avg
            info_dict['totalS_var'] = totalS_var

            SU2_prefactor = 1.
            Earray = Earray + SU2_prefactor * totalS_array

        ## [TODO] Add parse arg controll over whether adding chemical potential ?
        mu = 0.
        # localE_arr += mu * (num_particle != Lx*Ly//2)
        # localE_arr += mu * num_particle
        # localE_arr += mu * (num_particle - Lx*Ly//2)**2


        end_c, end_t = time.clock(), time.time()
        if verbose:
            print("monte carlo time ( localE ): ", end_c - start_c, end_t - start_t)

        Eavg = np.average(Earray)
        Evar = np.var(Earray)
        E0avg = np.average(E0array)
        E0var = np.var(E0array)

        amp_batch = self.get_self_amp_batch()
        max_amp = amp_batch[np.argmax(np.abs(amp_batch))]
        info_dict['max_amp'] = max_amp
        info_dict['E0'] = E0avg / num_site
        info_dict['E'] = Eavg / num_site
        info_dict['E0_var'] = Evar / (num_site**2)
        info_dict['E_var'] = E0var / (num_site**2)


        if verbose:
            print(amp_batch[:5])
            print("E/N !!!!: ", Eavg / num_site, "  Var: ", Evar / (num_site**2) )
            print("E0/N !!!!: ", E0avg / num_site, "  Var: ", E0var / (num_site**2) )

        if not SR:
            if self.moving_E_avg != None:
                self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
                print("moving_E_avg/N !!!!: ", self.moving_E_avg / num_site)
                Earray = Earray - self.moving_E_avg
            else:
                Earray = Earray - Eavg

            Glist = self.NNet.get_E_grads(config_arr, Earray)
            ## The below seems to be wrong.
            # Glist = self.NNet.get_E_grads(config_arr, Earray.conjugate())
            # Reg
            for idx, W in enumerate(self.NNet.sess.run(self.NNet.para_list)):
                Glist[idx] = Glist[idx] /num_sample + W * self.reg

            Gj = np.concatenate([g.flatten() for g in Glist])
            G_norm = np.linalg.norm(Gj)
            info_dict['G_norm'] = G_norm

            end_c, end_t = time.clock(), time.time()
            if verbose:
                print("monte carlo time ( back propagation to get E_grads ): ",
                      end_c - start_c, end_t - start_t)
                print("norm(G): ", G_norm)

            #
            # TO FIND BUG IN COMPLEX DERIVATIVE
            # COMPARING TO PLAIN GRADIENT
            # RESULT SHOW THAT WE DO NOT NEED TO SPLIT GRADIENT
            #
            # import pdb;pdb.set_trace()
            # Oarray = self.NNet.run_unaggregated_gradient(config_arr)
            # Osum = Oarray.conjugate().dot(np.ones(Oarray.shape[1]))  # <O*>
            # EOsum = Oarray.conjugate().dot(Earray)  # <O^*E>
            # Eavg = np.average(Earray)
            # Evar = np.var(Earray)
            # Fj = np.real( (EOsum / num_sample - Eavg * Osum / num_sample))
            # import matplotlib.pyplot as plt
            # print('gj dot fj', Gj.dot(Fj)/np.linalg.norm(Gj)/np.linalg.norm(Fj))
            # plt.plot(Gj/np.linalg.norm(Gj), label='Gj')
            # plt.plot(Fj/np.linalg.norm(Fj), label='Fj')
            # plt.legend()
            # plt.show()
            # import pdb;pdb.set_trace()
            return Gj, None, info_dict
        elif KFAC:  # SR + KFAC
            if self.moving_E_avg != None:
                self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
                print("moving_E_avg/N !!!!: ", self.moving_E_avg / num_site)
                Earray = Earray - self.moving_E_avg
            else:
                Earray = Earray - Eavg

            Glist = self.NNet.get_E_grads(config_arr, Earray)
            ## The below seems to be wrong.
            # Glist = self.NNet.get_E_grads(config_arr, Earray.conjugate())

            # Reg
            for idx, W in enumerate(self.NNet.sess.run(self.NNet.para_list)):
                Glist[idx] = Glist[idx] /num_sample + W * self.reg

            Gj = np.concatenate([g.flatten() for g in Glist])
            end_c, end_t = time.clock(), time.time()
            print("monte carlo time ( back propagation to get E_grads ): ",
                  end_c - start_c, end_t - start_t)
            print("norm(G): ", np.linalg.norm(Gj))

            F0_vec = Gj
            F_list = self.vec_to_list(F0_vec)

            # compute <O> (or <O^*>)??
            Olist = self.NNet.get_log_grads(config_arr)
            Oi = np.concatenate([g.flatten() for g in Olist]) / num_sample
            end_c, end_t = time.clock(), time.time()
            print("<O> time (batch_gradient): ", end_c - start_c, end_t - start_t)

            # Initiate KFAC
            self.NNet.apply_cov_update(config_arr)
            try:
                self.NNet.apply_inverse_update(config_arr)
            except Exception as e:
                print("not inverse upadte !!!")
                print(e)

            end_c, end_t = time.clock(), time.time()
            print("KFAC time ( OO update): ", end_c - start_c, end_t - start_t)
            Gj = self.list_to_vec([pair[0] for pair in self.NNet.apply_fisher_inverse(F_list, config_arr)])
            print("norm(G): ", np.linalg.norm(Gj))
            return Gj, Eavg / num_site, Evar / (num_site**2), Gj.dot(F0_vec)
            import pdb;pdb.set_trace()

            '''
            # compute OO, OO_F
            Oarray = self.NNet.run_unaggregated_gradient(config_arr)
            OOsum = Oarray.conjugate().dot(Oarray.T)
            OO = OOsum / num_sample
            end_c, end_t = time.clock(), time.time()
            print("OO explicit time ( SF ): ", end_c - start_c, end_t - start_t)

            # compute KFAC_OO_F
            KFAC_OO_F = self.NNet.apply_fisher_multiply(F_list, config_arr)
            KFAC_OO_F = self.list_to_vec([pair[0] for pair in KFAC_OO_F])
            end_c, end_t = time.clock(), time.time()
            print("KFAC time ( OO_F): ", end_c - start_c, end_t - start_t)

            # compute KFAC_OO
            FB_BLOCKS = list(self.NNet.layer_collection.get_blocks())
            OO_list = []
            for i in FB_BLOCKS:
                print(i._renorm_coeff)
                OO_list.append(self.NNet.sess.run(i.full_fisher_block()))

            KFAC_OO = scipy.linalg.block_diag(*OO_list)
            end_c, end_t = time.clock(), time.time()
            print("KFAC time ( construct OO): ", end_c - start_c, end_t - start_t)


            print("< KFAC OO_F , F > :",
                  KFAC_OO_F.dot(F0_vec)/np.linalg.norm(F0_vec)/np.linalg.norm(KFAC_OO_F))
            OO_F = OO.dot(F0_vec)
            print("< OO_F , F > :",
                  OO_F.dot(F0_vec)/np.linalg.norm(F0_vec)/np.linalg.norm(OO_F))
            print("< KFAC OO_F, OO_F > :",
                  OO_F.dot(KFAC_OO_F)/np.linalg.norm(OO_F)/np.linalg.norm(KFAC_OO_F))
            print(" norm (OO_F) / norm (multiply_F) ", np.linalg.norm(OO_F)/np.linalg.norm(KFAC_OO_F))

            import pdb;pdb.set_trace()

            # compute <O><O>
            O_O = np.einsum('i,j->ij', Oi.conjugate(), Oi)
            O_O_F = Oi.conjugate()*(Oi.dot(F0_vec))
            S_F = OO_F-O_O_F
            S_F_ = KFAC_OO_F-O_O_F
            print("< S_F, S_F_ > : ",
                  S_F.dot(S_F_)/np.linalg.norm(S_F)/np.linalg.norm(S_F_))
            print(" norm (S_F) / norm (S_F_) ", np.linalg.norm(S_F)/np.linalg.norm(S_F_))
            '''

            # ## PLOTTING THE OO_F and KFAC_OO_F vector
            # import matplotlib.pyplot as plt
            # plt.plot(OO_F/np.linalg.norm(OO_F), 'r'); plt.plot(KFAC_OO_F/np.linalg.norm(KFAC_OO_F), 'b'); plt.show();
            # import pdb;pdb.set_trace()


            # ## TESTING Fisher_inverse
            # invOO_F0 = np.linalg.pinv(OO, 1e-6).dot(F0_vec)
            # print(np.linalg.norm(invOO_F0), np.linalg.norm(G_prev_vec))
            # invOO_F0 /= np.linalg.norm(invOO_F0)
            # G_prev_vec /= np.linalg.norm(G_prev_vec)
            # print(invOO_F0.dot(G_prev_vec))

            # import matplotlib.pyplot as plt
            # plt.plot(G_prev_vec, 'r')
            # plt.plot(invOO_F0, 'b')
            # plt.show()


            def implicit_S(v):
                avgO = Oi
                finalv = - avgO.conjugate() * avgO.dot(v)

                v_list = self.vec_to_list(v)
                KFAC_OO_v = self.NNet.apply_fisher_multiply(v_list, config_arr)
                finalv += self.list_to_vec([pair[0] for pair in KFAC_OO_v])

                return np.real(finalv)  + v * 1e-4

            implicit_Sij = LinearOperator((numPara, numPara), matvec=implicit_S)

            Gj, info = scipy.sparse.linalg.minres(implicit_Sij, F0_vec, x0=Gj)

            end_c, end_t = time.clock(), time.time()
            print("solving SG=F: ", end_c - start_c, end_t - start_t)
            print("conv Gj : ", info)


            print("norm(G): ", np.linalg.norm(Gj),
                  "norm(F):", np.linalg.norm(F0_vec),
                  "G.dot(F):", Gj.dot(F0_vec))

            return Gj, Eavg / num_site, Evar / (num_site**2), Gj.dot(F0_vec)

        else: #if SR; elif KFAC; else
            if self.cov_list is None:
                self.cov_list = []
                for var_shape in self.NNet.var_shape_list:
                    var_size = np.prod(var_shape)
                    if var_size < 512:
                        self.cov_list.append(np.eye(var_size, dtype=self.NP_FLOAT))
                        # self.cov_list.append(np.zeros([var_size, var_size], dtype=self.NP_FLOAT))
                    else:
                        self.cov_list.append(None)

                # Now we have the cov_list setted up as a list of identity matrices
            else:
                pass


        Oarray = self.NNet.run_unaggregated_gradient(config_arr)
        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( back propagation to get log_grads ): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     O_List = self.NNet.get_log_grads(config_arr[i:i+1])
        #     Oarray[:, i] = np.concatenate([g.flatten() for g in O_List])

        # end_c, end_t = time.clock(), time.time()
        # print("monte carlo time ( back propagtation to get log_grads ): ", end_c - start_c, end_t - start_t)
        # print("difference in backprop : ", np.linalg.norm(Oarray2-Oarray))

        # Osum = np.einsum('ij->i', Oarray)
        # EOsum = np.einsum('ij,j->i', Oarray, Earray)
        Osum = Oarray.conjugate().dot(np.ones(Oarray.shape[1]))  # This < O^* > * num_sample
        ######### NEW MODIFICATION FOR KFAC like SR update scheme ##########
        EOsum = Oarray.conjugate().dot(Earray)  # <O^*E>
        ######### NEW MODIFICATION FOR KFAC like SR update scheme ##########

        ######### NEW MODIFICATION FOR KFAC like SR update scheme ##########
        #########               START                             ##########

        unaggregated_O_list = self.Oarray_to_Olist(Oarray - np.outer(Osum/num_sample, np.ones(num_sample)))
        ## The unaggregated_O has mean subtracted already
        decay_rate = 0.85
        new_k_size = 32
        save_k_size = 64
        stablize_eps = 1e-4  # at the level of eigenvalue, i.e. singular value ^ 2
        max_var_size = 8192
        for idx, unaggregated_O in enumerate(unaggregated_O_list):
            var_size = unaggregated_O.shape[0]
            if var_size < 512:
                ## Forming cov, inv_cov explicitly
                cov = (unaggregated_O.conjugate().dot(unaggregated_O.T) ).real / num_sample
                cov += np.eye(cov.shape[0]) * stablize_eps
                self.cov_list[idx] = self.cov_list[idx] * decay_rate + cov * (1-decay_rate)
            else:
                if var_size < max_var_size:
                    cov = (unaggregated_O.real.dot(unaggregated_O.real.T) ) / num_sample
                    cov += (unaggregated_O.imag.dot(unaggregated_O.imag.T)) / num_sample
                    # cov_U, cov_S, cov_Vd = scipy.sparse.linalg.svds(cov, k=k_size)
                    if self.cov_list[idx] is not None:
                        old_cov_S, old_cov_U = self.cov_list[idx][:]
                        cov = cov * (1-decay_rate) + old_cov_U.dot(np.diag(old_cov_S).dot(old_cov_U.T)) * decay_rate
                        try:
                            cov_S, cov_U = scipy.sparse.linalg.eigsh(cov, k=save_k_size)
                        except:
                            cov_S, cov_U = np.linalg.eigsh(cov)
                            # cov_S, cov_U = scipy.sparse.linalg.eigsh(cov, k=save_k_size//2)

                        self.cov_list[idx] = [cov_S, cov_U]
                    else:
                        cov_S, cov_U = scipy.sparse.linalg.eigsh(cov, k=new_k_size)
                        self.cov_list[idx] = [cov_S, cov_U]

                else:  # var_size > 8192
                    def implicit_cov(_v):
                        real_part = unaggregated_O.real
                        imag_part = unaggregated_O.imag
                        finalv = real_part.dot(real_part.T.dot(_v) / num_sample)
                        finalv += imag_part.dot(imag_part.T.dot(_v) / num_sample)
                        return finalv  + _v * stablize_eps

                    implicit_cov_op = LinearOperator((var_size, var_size), matvec=implicit_cov, rmatvec=implicit_cov)
                    try:
                        cov_S, cov_U = scipy.sparse.linalg.eigsh(implicit_cov_op, k=new_k_size)
                    except:
                        print("fail converge in eigsh")
                        cov_S, cov_U = scipy.sparse.linalg.eigsh(implicit_cov_op, k=new_k_size//2)

                    if self.cov_list[idx] is not None:
                        old_cov_S, old_cov_U = self.cov_list[idx][:]
                        def implicit_cov_all(_v):
                            finalv = cov_U.dot(np.diag(cov_S).dot(cov_U.T.dot(_v))) * (1-decay_rate)
                            finalv += old_cov_U.dot(np.diag(old_cov_S).dot(old_cov_U.T.dot(_v))) * (decay_rate)
                            return finalv

                        implicit_cov_all_op = LinearOperator((var_size, var_size), matvec=implicit_cov_all,
                                                             rmatvec=implicit_cov_all)
                        try:
                            cov_S, cov_U = scipy.sparse.linalg.eigsh(implicit_cov_all_op, k=save_k_size)
                        except:
                            cov_S, cov_U = scipy.sparse.linalg.eigsh(implicit_cov_all_op, k=old_cov_S.size+8)

                        self.cov_list[idx] = [cov_S, cov_U]
                    else:
                        self.cov_list[idx] = [cov_S, cov_U]


        # End for update cov_list
        end_c, end_t = time.clock(), time.time(); print("monte carlo time ( update cov_list ): ", end_c - start_c, end_t - start_t)

        if self.moving_E_avg != None:
            self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
            print("moving_E_avg/N !!!!: ", self.moving_E_avg / num_site)
            Earray_m_avg = Earray - self.moving_E_avg
        else:
            Earray_m_avg = Earray - Eavg

        _Flist = self.NNet.get_E_grads(config_arr, Earray_m_avg)


        end_c, end_t = time.clock(), time.time(); print("monte carlo time ( get E_grads ): ", end_c - start_c, end_t - start_t)

        ## Adding Regularization ##
        for idx, W in enumerate(self.NNet.sess.run(self.NNet.para_list)):
            _Flist[idx] = _Flist[idx] / num_sample + W * self.reg

        _Glist = []
        for idx, cov in enumerate(self.cov_list):
            if type(cov) != list:
                # inv_cov = np.linalg.pinv(cov, self.pinv_rcond)
                # _g = inv_cov.dot(_Flist[idx].flatten())
                # _Glist.append(_g)

                _g, info = scipy.sparse.linalg.minres(cov, _Flist[idx].flatten())
                if info != 0:
                    print(info)
                    import pdb;pdb.set_trace()
                else:
                    _Glist.append(_g)
            else:
                cov_S, cov_U = cov[:]
                # print(cov_S)
                # _g = cov_U.dot(np.diag(1./cov_S).dot(cov_U.T.dot(_Flist[idx].flatten())))
                cov_S[cov_S<0]=0
                _g = cov_U.dot(np.diag(cov_S/(cov_S**2 + stablize_eps)).dot(cov_U.T.dot(_Flist[idx].flatten())))
                _Glist.append(_g)


        end_c, end_t = time.clock(), time.time(); print("monte carlo time ( inverse cov_list get G ): ", end_c - start_c, end_t - start_t)


        _Fj = np.concatenate([_f.flatten() for _f in _Flist])
        _Gj = np.concatenate([_g.flatten() for _g in _Glist])
        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( compute inv OO from cov_list and get Gj ): ",
              end_c - start_c, end_t - start_t)

        _GjFj = np.linalg.norm(_Gj.dot(_Fj))
        print("norm(G): ", np.linalg.norm(_Gj),
              "norm(F):", np.linalg.norm(_Fj),
              "G.dot(F):", _GjFj)
        # import pdb;pdb.set_trace()
        return _Gj, Eavg / num_site, Evar / (num_site**2) / num_sample, _GjFj


        ######### NEW MODIFICATION FOR KFAC like SR update scheme ##########
        #########               END                               ##########
        ####################################################################


        if not explicit_SR:
            pass
        else:
            # One of the expressions below should be wrong and
            # should be modified.
            # (1.)
            OOsum = Oarray.conjugate().dot(Oarray.T)
            # (2.)
            # mask = 1 - 2*self.NNet.im_para_array
            # Oarray_ = np.einsum('i,ij->ij', mask, Oarray)
            # OOsum = Oarray.dot(Oarray_.T)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (total): ", end_c - start_c, end_t - start_t)
        start_c, start_t = time.clock(), time.time()

        #####################################
        #  Fj = 2Re[ <O_iH> - <H><O_i> ]
        #  Fj = <O_iH> - <H><O_i>
        #####################################

        # The networks are all parametrized by real-valued variables.
        # As a result, Fj should be real and is real by definition.
        # we cast the type to real by np.real

        if self.moving_E_avg is None:
            # Fj = np.real(2. * (EOsum / num_sample - Eavg * Osum / num_sample))
            Fj = np.real((EOsum / num_sample - Eavg * Osum / num_sample))
            Fj += self.reg * np.concatenate([g.flatten() for g in self.NNet.sess.run(self.NNet.para_list)])
            # if self.using_complex:
            #     Fj = Fj[self.re_idx_array] + 1j*Fj[self.im_idx_array]
            # else:
            #     pass
        else:
            self.moving_E_avg = self.moving_E_avg * 0.5 + Eavg * 0.5
            # Fj = np.real(2. * (EOsum / num_sample - self.moving_E_avg * Osum / num_sample))
            Fj = np.real((EOsum / num_sample - self.moving_E_avg * Osum / num_sample))
            print("moving_E_avg/N !!!!: ", self.moving_E_avg / num_site)

        if not explicit_SR:
            def implicit_S(v):
                avgO = Osum.flatten()/num_sample
                # finalv = - avgO.dot(v) * avgO.conjugate()
                finalv = - avgO.conjugate() * avgO.dot(v)
                finalv += Oarray.conjugate().dot((Oarray.T.dot(v)))/num_sample
                return np.real(finalv)  # + v * 1e-4

            implicit_Sij = LinearOperator((numPara, numPara), matvec=implicit_S)

            Gj, info = scipy.sparse.linalg.minres(implicit_Sij, Fj, x0=Gj)
            print("conv Gj : ", info)
        else:
            #####################################
            # S_ij = <O_i O_j > - <O_i><O_j>   ##
            #####################################
            # There are two formulation here and one should be correct.
            # (1.)
            # why no conjugate here ?
            Sij = np.real(OOsum / num_sample - np.einsum('i,j->ij', Osum.flatten(), Osum.flatten()) / (num_sample**2))

            # (2.)
            # Sij = OOsum / num_sample - np.einsum('i,j->ij', Osum.flatten() / num_sample,
            #                                      np.einsum('i,i->i', Osum.flatten()/num_sample,
            #                                                (1-2*self.NNet.im_para_array)))
            # if self.using_complex:
            #     Sij = (Sij[self.re_idx_array][:, self.re_idx_array] -
            #            Sij[self.im_idx_array][:, self.im_idx_array] +
            #            1j * Sij[self.im_idx_array][:, self.re_idx_array] +
            #            1j * Sij[self.re_idx_array][:, self.im_idx_array])


            # Adding regularization/dumping/rotation 
            # regu_para = np.amax([10 * (0.9**iteridx), 1e-4])
            # Sij = Sij + regu_para * np.diag(np.ones(Sij.shape[0]))
            if not self.real_time:
                Sij = Sij+np.diag(np.ones(Sij.shape[0])*1e-4)
            else:
                pass
            ############
            # Method 1 #
            ############
            # invSij = np.linalg.inv(Sij)
            # Gj = invSij.dot(Fj.T)
            ############
            # Method 2 #
            ############
            if self.real_time:
                # Evar_ = (Evar / (L**2) / num_sample)
                invSij = np.linalg.pinv(Sij, self.pinv_rcond)
                Gj = invSij.dot(Fj.T)
            ############
            # Method 3 #
            ############
            else:
                # possible method, minres, lgmres, cg
                Gj, info = scipy.sparse.linalg.minres(Sij, Fj, x0=Gj)
                # Gj, info = scipy.sparse.linalg.cg(Sij, Fj)  # , x0=Gj)
                print("conv Gj : ", info)

        # Gj = Fj.T
        GjFj = np.linalg.norm(Gj.dot(Fj))
        print("norm(G): ", np.linalg.norm(Gj),
              "norm(F):", np.linalg.norm(Fj),
              "G.dot(F):", GjFj)

        end_c, end_t = time.clock(), time.time()
        print("Sij, Fj time: ", end_c - start_c, end_t - start_t)

        # (2.)
        # if self.using_complex:
        #     tmp = np.zeros(Sij.shape[0]*2)
        #     tmp[self.re_idx_array] = np.real(Gj)
        #     tmp[self.im_idx_array] = np.imag(Gj)
        #     Gj = tmp

        if self.real_time:
            tmp = Gj[self.re_idx_array]
            Gj[self.re_idx_array] = -Gj[self.im_idx_array]
            Gj[self.im_idx_array] = tmp

        import pdb;pdb.set_trace()
        print(" TO debug FjFj, ", Fj.dot(_Fj)/np.linalg.norm(Fj)/np.linalg.norm(_Fj),
              " TO debug GjGj, ", Gj.dot(_Gj)/np.linalg.norm(Gj)/np.linalg.norm(_Gj))
        return Gj, Eavg / num_site, Evar / (num_site**2) / num_sample, GjFj


class NQS_1d(NQS_base):
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1, J2=None, reg=0.,
                 using_complex=False, single_precision=True, real_time=False,
                 pinv_rcond=1e-6, PBC=False):
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1]),
                               dtype=np.int8)
        self.num_site = inputShape[0]
        self.channels = inputShape[1]
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.init_config(sz0_sector=True)
        self.corrlength = inputShape[0]
        self.max_batch_size = 1024
        if self.batch_size > self.max_batch_size:
            print("batch_size > max_batch_size, memory error may occur")

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None
        self.reg = reg
        self.using_complex = using_complex
        self.SP = single_precision
        if self.SP:
            self.NP_FLOAT = np.float32
            self.NP_COMPLEX = np.complex64
        else:
            self.NP_FLOAT = np.float64
            self.NP_COMPLEX = np.complex128

        self.PBC = PBC
        self.real_time = real_time
        self.pinv_rcond = pinv_rcond
        np_arange = np.arange(self.net_num_para)
        self.re_idx_array = np_arange[np.array(1-self.NNet.im_para_array,
                                               dtype=bool)]
        self.im_idx_array = np_arange[np.array(self.NNet.im_para_array,
                                               dtype=bool)]

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            self.get_local_E_batch = self.local_E_Ising_batch
            self.new_config_batch = self.new_config_batch_single
        elif Hamiltonian == 'Sz':
            self.get_local_E_batch = self.local_E_Sz_batch
            self.new_config_batch = self.new_config_batch_single
        elif Hamiltonian == 'AFH':
            self.get_local_E_batch = self.local_E_AFH_batch
            self.new_config_batch = self.new_config_batch_sz
        elif Hamiltonian == 'J1J2':
            self.J2 = J2
            self.get_local_E_batch = self.local_E_J1J2_batch
            self.new_config_batch = self.new_config_batch_sz
            # self.get_local_E_batch = self.local_E_J1J2_batch_log
        else:
            raise NotImplementedError

        super(NQS_1d, self).__init__()

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.inputShape[0]))
                while(np.sum(x) != self.inputShape[0]//2):
                    x = np.random.randint(2, size=(self.inputShape[0]))

                self.config[i, :, 0] = x
                self.config[i, :, 1] = (1 - x)

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.inputShape[0]))
            self.config[:, :, 0] = x
            self.config[:, :, 1] = (1 - x)
            return

    def new_config(self):
        L = self.config.shape[1]

        # Restricted to Sz = 0 sectors ##
        randsite1 = np.random.randint(L)
        randsite2 = np.random.randint(L)
        if self.config[0, randsite1, 0] + self.config[0, randsite2, 0] == 1 and randsite1 != randsite2:
            tempconfig = self.config.copy()
            tempconfig[0, randsite1, :] = (1 - tempconfig[0, randsite1, :])
            tempconfig[0, randsite2, :] = (1 - tempconfig[0, randsite2, :])
            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
            if np.random.rand() < np.amin([1., ratio**2]):
                self.config = tempconfig
            else:
                pass
        else:
            pass


#        tempconfig = self.config.copy()
#        if np.random.rand() < 0.5:
#            randsite = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
#        else:
#            randsite = np.random.randint(L)
#            randsite2 = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            tempconfig[0, randsite2, :] = (tempconfig[0, randsite2, :] + 1) % 2
#            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
#            if np.random.rand() < np.amin([1., ratio**2]):
#                self.config = tempconfig
#            else:
#                pass

        return

    def new_config_batch_sz(self):
        L = self.config.shape[1]
        batch_size = self.batch_size
        old_log_amp = self.get_self_log_amp_batch()
        # old_amp = self.get_self_amp_batch()

        # Restricted to Sz = 0 sectors ##
        randsite1 = np.random.randint(L, size=(batch_size,))
        randsite2 = np.random.randint(L, size=(batch_size,))
        mask = (self.config[np.arange(batch_size), randsite1, 0] +
                self.config[np.arange(batch_size), randsite2, 0]) == 1

        flip_config = self.config.copy()
        flip_config[np.arange(batch_size), randsite1, :] = (1 - flip_config[np.arange(batch_size), randsite1, :])
        flip_config[np.arange(batch_size), randsite2, :] = (1 - flip_config[np.arange(batch_size), randsite2, :])

        # ratio_square = np.zeros((batch_size,))
        # ratio_square[mask] = np.exp(2.*np.real(self.eval_log_amp_array(flip_config[mask]) - old_log_amp[mask]))
        # # ratio = np.divide(np.abs(self.eval_amp_array(flip_config))+1e-45, np.abs(old_amp)+1e-45 )
        # # ratio_square = np.power(ratio,  2)
        # mask2 = np.random.random_sample((batch_size,)) < ratio_square

        ratio_log = -np.inf * np.ones((batch_size,))
        ratio_log[mask] = 2.*np.real(self.eval_log_amp_array(flip_config[mask]) - old_log_amp[mask])
        mask2 = np.log(np.random.random_sample((batch_size,))) < ratio_log

        final_mask = np.logical_and(mask, mask2)
        # update self.config
        self.config[final_mask] = flip_config[final_mask]

        acceptance_ratio = np.sum(final_mask)/batch_size
        return acceptance_ratio

    def new_config_batch_single(self):
        L = self.config.shape[1]
        batch_size = self.batch_size
        old_log_amp = self.get_self_log_amp_batch()
        # old_amp = self.get_self_amp_batch()

        randsite1 = np.random.randint(L, size=(batch_size,))

        flip_config = self.config.copy()
        flip_config[np.arange(batch_size), randsite1, :] = (1 - flip_config[np.arange(batch_size), randsite1, :])

        ratio = np.zeros((batch_size,))
        ratio = np.exp(2.*np.real(self.eval_log_amp_array(flip_config) - old_log_amp))
        ratio_square = ratio
        # ratio = np.divide(np.abs(self.eval_amp_array(flip_config))+1e-45, np.abs(old_amp)+1e-45 )
        # ratio_square = np.power(ratio,  2)
        mask2 = np.random.random_sample((batch_size,)) < ratio_square

        final_mask = mask2
        # update self.config
        self.config[final_mask] = flip_config[final_mask]

        acceptance_ratio = np.sum(final_mask)/batch_size
        return acceptance_ratio

    def H_exp(self, num_sample=1000, h=1, J=0):
        energyList = []
        correlength = 10
        for i in range(num_sample * correlength):
            self.new_config()
            if i % correlength == 0:
                _, _, localEnergy, _ = self.getLocal(self.config)
                energyList = np.append(energyList, localEnergy)
            else:
                pass
        print(energyList)
        return np.average(energyList)

    def VMC_observable(self, num_sample):
        L = self.config.shape[1]
        numPara = self.net_num_para

        start_c, start_t = time.clock(), time.time()
        corrlength = self.corrlength
        configDim = list(self.config.shape)
        configDim[0] = num_sample
        config_arr = np.zeros(configDim, dtype=np.int32)

        if (self.batch_size == 1):
            for i in range(1, 1 + num_sample * corrlength):
                self.new_config()
                if i % corrlength == 0:
                    config_arr[i // corrlength - 1, :, :] = self.config[0, :, :]

        else:
            for i in range(1, 1 + (num_sample * corrlength) // self.batch_size):
                self.new_config_batch()
                bs = self.batch_size
                if i % corrlength == 0:
                    i_c = i // corrlength
                    config_arr[(i_c-1)*bs: i_c*bs, :, :] = self.config[:, :, :]
                else:
                    pass

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time (gen config): ", end_c - start_c, end_t - start_t)

        # for i in range(num_sample):
        #     Earray[i] = self.get_local_E(config_arr[i:i+1])

        SzSz = self.sz_sz_expectation(config_arr)
        Sz = self.sz_expectation(config_arr)
        local_E = self.Ising_local_expectation(config_arr)

        end_c, end_t = time.clock(), time.time()
        print("monte carlo time ( spin-spin-correlation ): ", end_c - start_c, end_t - start_t)
        return {"SzSz" : SzSz, "Sz": Sz, "local_E": local_E}

    def getLocal_no_OO(self, config):
        '''
        forming OO is extremely slow.
        test with np.einsum, np.outer
        '''
        localE = self.get_local_E(config)
        # localE2 = self.local_E_AFH_old(config)
        # if (localE-localE2)>1e-12:
        #     print(np.squeeze(config).T, localE, localE2)

        O_List = self.NNet.get_log_grads(config)
        localO = np.concatenate([g.flatten() for g in O_List])
        localEO = localO * localE

        return localO, localE, localEO

    # local_E_Ising, get local E from Ising Hamiltonian
    # For only one config.
    def local_E_Ising(self, config, h=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        for i in range(L - 1):
            temp = config[0, i, :].dot(config[0, i + 1, :])
            localE -= 2 * (temp - 0.5)

        # Periodic Boundary condition
        temp = config[0, 0, :].dot(config[0, -1, :])
        localE -= 2 * (temp - 0.5)
        #####################################

        oldAmp = self.eval_amp_array(config)[0]
        for i in range(L):
            tempConfig = config.copy()
            tempConfig[0, i, :] = (1 - tempConfig[0, i, :])
            tempAmp = float(self.NNet.get_amp(tempConfig))
            localE -= h * tempAmp / oldAmp

        return localE

    def local_E_AFH_old(self, config, J=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        oldAmp = self.eval_amp_array(config)[0]

        for i in range(L - 1):
            # Sz Sz Interaction
            temp = config[0, i, :].dot(config[0, i + 1, :])
            localE += 2 * (temp - 0.5) * J / 4
            if config[0, i, :].dot(config[0, i + 1, :]) == 0:
                tempConfig = config.copy()
                tempConfig[0, i, :] = (1 - tempConfig[0, i, :])
                tempConfig[0, i + 1, :] = (1 - tempConfig[0, i + 1, :])
                tempAmp = float(self.NNet.get_amp(tempConfig))
                localE += J * tempAmp / oldAmp / 2
            else:
                pass
        '''
        Periodic Boundary condition
        '''
        temp = config[0, 0, :].dot(config[0, -1, :])
        localE += 2 * (temp - 0.5) * J / 4
        if temp == 0:
            tempConfig = config.copy()
            tempConfig[0, 0, :] = (1 - tempConfig[0, 0, :])
            tempConfig[0, L-1, :] = (1 - tempConfig[0, L-1, :])
            tempAmp = float(self.NNet.get_amp(tempConfig))
            localE += J * tempAmp / oldAmp / 2

        return localE

    def local_E_AFH(self, config, J=1):
        numData, L, inputShape1 = config.shape
        localE = 0.
        oldAmp = self.eval_amp_array(config)[0]

        # PBC
        config_shift_copy = np.zeros((1, L, inputShape1), dtype=np.int32)
        config_shift_copy[:, :-1, :] = config[:, 1:, :]
        config_shift_copy[:, -1, :] = config[:, 0, :]

        '''
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        SzSz = np.einsum('ij,ij->i', config[0, :, :], config_shift_copy[0, :, :])
        localE += np.sum(SzSz - 0.5) * 2 * J / 4

        config_flip = np.einsum('i,ijk->ijk', np.ones(L, dtype=np.int32), config)
        for i in range(L):
            config_flip[i, i, :] = (1 - config_flip[i, i, :])
            config_flip[i, (i+1) % L, :] = (1 - config_flip[i, (i+1) % L, :])

#        for i in range(L-1):
#            config_flip[i, i, :] = (config_flip[i, i, :] + 1) % 2
#            config_flip[i, (i+1), :] = (config_flip[i, (i+1), :] + 1) % 2

#        config_flip[L-1, 0, :] = (config_flip[L-1, 0, :] + 1) % 2
#        config_flip[L-1, L-1, :] = (config_flip[L-1, L-1, :] + 1) % 2

        flip_Amp = self.eval_amp_array(config_flip)
        localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2

        return localE

    def sz_sz_expectation(self, config_arr):
        '''
        Compute the spin-spin correlation <S^z_i S^z_j>
        with average over i
        '''
        num_config, L, inputShape1 = config_arr.shape

        config_shift_copy = np.zeros((num_config, L, inputShape1), dtype=np.int32)
        SzSz_j = [0.25]
        for j in range(1,L):
            config_shift_copy[:, :-j, :] = config_arr[:, j:, :]
            config_shift_copy[:, -j:, :] = config_arr[:, :j, :]
            SzSz = np.einsum('ijk,ijk->', config_arr, config_shift_copy) / L / num_config
            # SzSz average over L and num_config
            # In this convention SzSz = 1 or 0
            SzSz_j.append((SzSz-0.5)/2.)

        return(np.array(SzSz_j))

    def sz_expectation(self, config_arr):
        '''
        Compute the spin expectation <S^z_i>
        Assume [:,:,0] represent up
        '''
        num_config, L, inputShape1 = config_arr.shape

        Sz_j = np.einsum('ij->j', config_arr[:,:,0]) / num_config
        # Sz_j average over num_config
        # In this convention Sz_j is in [0,1]
        Sz_j = (Sz_j-0.5)
        return Sz_j

    def Ising_local_expectation(self, config_arr):
        '''
        H = -J sz_i sz_j + g sx_i - h sz_i
        '''
        J = 0.4
        g = 0.9045
        h = 0.7090
        PBC = self.PBC

        num_config, L, inputShape1 = config_arr.shape
        localE_arr = np.zeros(L, dtype=self.NP_COMPLEX)
        oldAmp = self.eval_amp_array(config_arr)

        ###########################
        #  J sigma^z_i sigma^z_j  #
        ###########################
        config_shift_copy = np.zeros((num_config, L, inputShape1), dtype=np.int32)
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # SzSz stores L : szsz expectation value over (i, i+1)
        # SzSz_ stores L : szsz expectation value over (i-1, i)
        SzSz = np.einsum('ijk,ijk->j', config_arr, config_shift_copy) / num_config
        SzSz_ = SzSz.copy()
        SzSz_[1:] = SzSz[:-1]
        SzSz_[0] = SzSz[-1]
        if not PBC:
            SzSz_[0] = 0.5  # 0.5 corresponds to E=0
            SzSz[-1] = 0.5  # 0.5 corresponds to E=0

        localE_arr += (SzSz - 0.5) * 2 * (-J)/2
        localE_arr += (SzSz_ - 0.5) * 2 * (-J)/2

        #################
        #  g sigma^x_i  #
        #################

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L, dtype=np.float32), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += g (flip_Amp) / oldAmp
        localE_arr += np.einsum('ij, j -> i',  flip_Amp_arr, 1./oldAmp * g) / num_config

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        Sz = config_arr[:, :, 0]
        localE_arr += ((np.einsum('ij->j', Sz) / num_config) - 0.5 ) * 2 * (-h)

        return localE_arr

    def local_E_AFH_batch(self, config_arr, J=1):
        '''
        Base on the fact that, in one-hot representation
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, L, inputShape1 = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        # PBC
        config_shift_copy = np.zeros((num_config, L, inputShape1), dtype=np.int32)
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L, dtype=np.int32), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+1) % L, :] = (1 - config_flip_arr[i, :, (i+1) % L, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J / oldAmp / 2
        return localE_arr

    def local_E_J1J2_batch(self, config_arr):
        J1 = 1.
        J2 = self.J2
        '''
        Base on the fact that, in one-hot representation
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        '''
        num_config, L, inputShape1 = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        ####################
        # PBC   J1 term   ##
        ####################
        config_shift_copy = np.zeros((num_config, L, inputShape1), dtype=np.int32)
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J1 / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L, dtype=np.int32), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+1) % L, :] = (1 - config_flip_arr[i, :, (i+1) % L, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J1 / oldAmp / 2

        ######################
        # PBC  J2 term      ##
        ######################
        config_shift_copy[:, :-2, :] = config_arr[:, 2:, :]
        config_shift_copy[:, -2:, :] = config_arr[:, :2, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * J2 / 4

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L, dtype=np.int32), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])
            config_flip_arr[i, :, (i+2) % L, :] = (1 - config_flip_arr[i, :, (i+2) % L, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += -(SzSz-1).dot(flip_Amp) * J / oldAmp / 2
        localE_arr += -np.einsum('ij,ji->i', (SzSz-1), flip_Amp_arr) * J2 / oldAmp / 2
        return localE_arr

    def local_E_Ising_batch(self, config_arr):
        '''
        https://arxiv.org/pdf/1503.04508.pdf
        H = -J sx_i sx_j - g sz_i - h sx_i

        for convenient we use the following convention
        H = -J sz_i sz_j + g sx_i - h sz_i
        '''
        J = 0.4 # self.J
        g = 0.9045 # self.g
        h = 0.7090 # self.h
        PBC = self.PBC
        '''
        Base on the fact that, in one-hot representation
        sigma^z siamg^z Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du

        we assume 0 represent up and 1 represent down
        '''
        J = 1 # self.J
        g = 3.5 # self.g
        h = 0. # self.h
        num_config, L, inputShape1 = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        ###########################
        #  J sigma^z_i sigma^z_j  #
        ###########################
        config_shift_copy = np.zeros((num_config, L, inputShape1), dtype=np.int32)
        config_shift_copy[:, :-1, :] = config_arr[:, 1:, :]
        config_shift_copy[:, -1, :] = config_arr[:, 0, :]

        # num_config x L
        SzSz = np.einsum('ijk,ijk->ij', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ij->i', SzSz - 0.5) * 2 * (-J)
        else:
            localE_arr += np.einsum('ij->i', SzSz[:,:-1] - 0.5) * 2 * (-J)

        #################
        #  g sigma^x_i  #
        #################

        # num_site(L) x num_config x num_site(L) x num_spin
        config_flip_arr = np.einsum('h,ijk->hijk', np.ones(L, dtype=np.float32), config_arr)
        for i in range(L):
            config_flip_arr[i, :, i, :] = (1 - config_flip_arr[i, :, i, :])

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(L*num_config, L, inputShape1))
        flip_Amp_arr = flip_Amp_arr.reshape((L, num_config))
        # localE += g (flip_Amp) / oldAmp
        localE_arr += np.einsum('ij -> j',  flip_Amp_arr) / oldAmp * g

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        Sz = (config_arr[:, :, 0] - 0.5) * 2
        localE_arr += np.einsum('ij->i', -Sz * h)

        return localE_arr

    def local_E_Sz_batch(self, config_arr):
        '''
        Because we can not apply the S+ directly to the state,
        We minimized Sz directly for the center site.
        This effectively leads to exp(-Sz) on the center site.
        we assume 0 represent up and 1 represent down.
        '''
        num_config, L, inputShape1 = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        Sz_mid = (config_arr[:, L//2, 0]-0.5)*2
        localE_arr += Sz_mid

        return localE_arr

    def local_E_J1J2_batch_log(self, config_arr):
        return NotImplementedError

############################
#  END OF DEFINITION NQS1d #
############################


class NQS_2d(NQS_base):
    def __init__(self, inputShape, Net, Hamiltonian, batch_size=1, J2=None, reg=0.,
                 using_complex=False, single_precision=True, real_time=False,
                 pinv_rcond=1e-6, PBC=False):
        '''
        config = [batch_size, Lx, Ly, local_dim]
        config represent the product state basis of the model
        in one-hot representation
        Spin-1/2 model: local_dim = 2
        Hubbard model: local_dim = 4
        '''
        self.config = np.zeros((batch_size, inputShape[0], inputShape[1], inputShape[2]),
                               dtype=np.int8)
        self.batch_size = batch_size
        self.inputShape = inputShape
        self.Lx = inputShape[0]
        self.Ly = inputShape[1]
        self.channels = inputShape[2]
        self.LxLy = self.num_site = self.Lx*self.Ly
        if self.Lx != self.Ly:
            print("not a square lattice !!!")

        self.init_config(sz0_sector=True)
        self.corrlength = self.LxLy
        self.max_batch_size = 1024
        if self.batch_size > self.max_batch_size:
            print("batch_size > max_batch_size, memory error may occur")

        self.NNet = Net
        self.net_num_para = self.NNet.getNumPara()
        self.moving_E_avg = None
        self.reg = reg
        self.using_complex = using_complex
        self.SP = single_precision
        if self.SP:
            self.NP_FLOAT = np.float32
            self.NP_COMPLEX = np.complex64
        else:
            self.NP_FLOAT = np.float64
            self.NP_COMPLEX = np.complex128

        self.PBC = PBC
        self.real_time = real_time
        self.pinv_rcond = pinv_rcond
        np_arange = np.arange(self.NNet.im_para_array.size)
        self.re_idx_array = np_arange[np.array((1-self.NNet.im_para_array), dtype=bool)]
        self.im_idx_array = np_arange[np.array(self.NNet.im_para_array, dtype=bool)]

        print("This NQS is aimed for ground state of %s Hamiltonian" % Hamiltonian)
        if Hamiltonian == 'Ising':
            self.get_local_E_batch = self.local_E_Ising_batch_log
            # self.get_local_E_batch = self.local_E_Ising_batch
        elif Hamiltonian == 'Sz':
            raise NotImplementedError
            '''
            should add one-site update scheme, so not restricted to
            sz=0 sector.
            also applicable to Ising model.
            '''
        elif Hamiltonian == 'AFH':
            # self.get_local_E_batch = self.local_E_2dAFH_batch
            self.get_local_E_batch = self.local_E_2dAFH_batch_log
        elif Hamiltonian == 'J1J2':
            self.J2 = J2
            # self.get_local_E_batch = self.local_E_2dJ1J2_batch
            self.get_local_E_batch = self.local_E_2dJ1J2_batch_log
        elif Hamiltonian == 'Julian':
            self.get_local_E_batch = self.local_E_2dJulian_batch_log
        else:
            raise NotImplementedError

        super(NQS_2d, self).__init__()

    def init_config(self, sz0_sector=True):
        if sz0_sector:
            for i in range(self.batch_size):
                x = np.random.randint(2, size=(self.Lx, self.Ly))
                while(np.sum(x) != self.LxLy//2):
                    x = np.random.randint(2, size=(self.Lx, self.Ly))

                self.config[i, :, :, 0] = x
                self.config[i, :, :, 1] = (1 - x)

            return
        else:
            x = np.random.randint(2, size=(self.batch_size, self.Lx, self.Ly))
            self.config[:, :, :, 0] = x
            self.config[:, :, :, 1] = (1 - x)
            return

    def new_config(self):
        '''
        Implementation for
        1.) random swap transition in spin-1/2 model
        2.) Restricted to Sz = 0 sectors
        '''
        randsite1_x = np.random.randint(self.Lx)
        randsite1_y = np.random.randint(self.Ly)
        randsite2_x = np.random.randint(self.Lx)
        randsite2_y = np.random.randint(self.Ly)
        cond1 = (self.config[0, randsite1_x, randsite1_y, 0] +
                 self.config[0, randsite2_x, randsite2_y, 0]) == 1
        cond2 = randsite1_x != randsite2_x
        cond3 = randsite1_y != randsite2_y
        if cond1 and cond2 and cond3:
            tempconfig = self.config.copy()
            tempconfig[0, randsite1_x, randsite1_y, :] = (1 - tempconfig[0, randsite1_x, randsite1_y, :])
            tempconfig[0, randsite2_x, randsite2_y, :] = (1 - tempconfig[0, randsite2_x, randsite2_y, :])
            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
            if np.random.rand() < np.amin([1., ratio**2]):
                self.config = tempconfig
            else:
                pass
        else:
            pass

#        tempconfig = self.config.copy()
#        if np.random.rand() < 0.5:
#            randsite = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
#        else:
#            randsite = np.random.randint(L)
#            randsite2 = np.random.randint(L)
#            tempconfig[0, randsite, :] = (tempconfig[0, randsite, :] + 1) % 2
#            tempconfig[0, randsite2, :] = (tempconfig[0, randsite2, :] + 1) % 2
#            ratio = self.NNet.get_amp(tempconfig)[0] / self.getSelfAmp()
#            if np.random.rand() < np.amin([1., ratio**2]):
#                self.config = tempconfig
#            else:
#                pass

        return

    def forward_sampling(self, sym_sec=None):
        '''
        Apply foward_sampling
        '''
        batch_size = self.batch_size
        ## Reset the config to zeros
        self.config = 0 * self.config
        if sym_sec is None:
            self.config = 0 * self.config
            for site_i in range(self.Lx):
                for site_j in range(self.Ly):
                    cond_prob_amp = self.NNet.plain_get_cond_log_amp(self.config)
                    # cond_prob_amp = cond_prob_amp.reshape([self.batch_size, *self.inputShape])
                    # site_cond_prob_amp = cond_prob_amp[:, site_i, site_j, :]
                    # mask = np.log(np.random.random_sample((batch_size,))) < site_cond_prob_amp[:,0]*2

                    cond_prob = np.exp(2 * cond_prob_amp.real)  # of shape [n_batch,...]
                    cond_prob = cond_prob.reshape([self.batch_size, *self.inputShape])
                    site_prob = cond_prob[:, site_i, site_j, :]
                    if self.channels == 2:
                        mask = np.random.random_sample((batch_size,)) < site_prob[:,0]
                        self.config[mask, site_i, site_j, 0] = 1
                        self.config[np.logical_not(mask), site_i, site_j, 1] = 1
                    else:
                        for batch_idx in range(batch_size):
                            self.config[batch_idx, site_i, site_j,
                                        np.random.choice(self.channels, p=site_prob[batch_idx])] = 1

                    assert( not np.isnan(site_prob).any() )


            return
        else:
            tmp_config = self.config.copy()
            num_sampled = 0
            while ( num_sampled < batch_size ):
                ## Start sampling
                self.config = 0 * self.config
                for site_i in range(self.Lx):
                    for site_j in range(self.Ly):
                        cond_prob_amp = self.NNet.plain_get_cond_log_amp(self.config)
                        cond_prob = np.exp(2 * cond_prob_amp.real)  # of shape [n_batch, 
                        cond_prob = cond_prob.reshape([self.batch_size, *self.inputShape])
                        site_prob = cond_prob[:, site_i, site_j, :]
                        mask = np.random.random_sample((batch_size,)) < site_prob[:,0]
                        self.config[mask, site_i, site_j, 0] = 1
                        self.config[np.logical_not(mask), site_i, site_j, 1] = 1

                Sz_array = np.sum(self.config[:,:,:,0], axis=(1,2))
                sym_mask = (Sz_array == self.Lx * self.Ly // 2)
                new_n_sampled = np.sum(sym_mask)
                if num_sampled+new_n_sampled <= batch_size:
                    tmp_config[num_sampled:num_sampled+new_n_sampled] = self.config[sym_mask]
                    num_sampled += new_n_sampled
                else:
                    tmp_config[num_sampled:] = self.config[sym_mask][:batch_size-num_sampled]
                    num_sampled = batch_size

            self.config = tmp_config.copy()
            return

    def new_config_batch(self):
        '''
        Implementation for
        1.) random swap transition in spin-1/2 model
        2.) Restricted to Sz = 0 sectors
        3.) vectorized for batch update
        4.) 10% propasal include global spin inversion
        '''
        ## Should add implementation with log amplitude
        ## which would be more stable for numerical reason.
        batch_size = self.batch_size
        old_log_amp = self.get_self_log_amp_batch()
        # old_amp = self.get_self_amp_batch()

        # Restricted to Sz = 0 sectors ##
        randsite1_x = np.random.randint(self.Lx, size=(batch_size,))
        randsite1_y = np.random.randint(self.Ly, size=(batch_size,))
        # Random Update
        # randsite2_x = np.random.randint(self.Lx, size=(batch_size,))
        # randsite2_y = np.random.randint(self.Ly, size=(batch_size,))
        # Local Update, (0) right (1) upper right (2) up 
        rand_direct = np.random.randint(3, size=(batch_size,))
        rand_dx = 1 - (rand_direct // 2)
        rand_dy = (rand_direct + 1) // 2
        randsite2_x = np.array( (randsite1_x + self.Lx + rand_dx) % self.Lx, dtype=np.int32)
        randsite2_y = np.array( (randsite1_y + self.Ly + rand_dy) % self.Ly, dtype=np.int32)

        mask = (self.config[np.arange(batch_size), randsite1_x, randsite1_y, 0] +
                self.config[np.arange(batch_size), randsite2_x, randsite2_y, 0]) == 1

        flip_config = self.config.copy()
        flip_config[np.arange(batch_size), randsite1_x, randsite1_y, :] = 1 - flip_config[np.arange(batch_size), randsite1_x, randsite1_y, :]
        flip_config[np.arange(batch_size), randsite2_x, randsite2_y, :] = 1 - flip_config[np.arange(batch_size), randsite2_x, randsite2_y, :]

        # Random total spin flip 10 % of configurations
        to_flip_idx = np.random.choice(batch_size, batch_size//10, replace=False)
        flip_config[to_flip_idx] = 1 - flip_config[to_flip_idx]

        # ratio_square = np.zeros((batch_size,))
        # ratio_square[mask] = np.exp(2.*np.real(self.eval_log_amp_array(flip_config[mask]) - old_log_amp[mask]))
        # # ratio[mask] = np.divide(np.abs(self.eval_amp_array(flip_config[mask]))+1e-45, np.abs(old_amp[mask])+1e-45 )
        # mask2 = np.random.random_sample((batch_size,)) < ratio_square

        ratio_log = -np.inf * np.ones((batch_size,))
        ratio_log[mask] = 2.*np.real(self.eval_log_amp_array(flip_config[mask]) - old_log_amp[mask])
        mask2 = np.log(np.random.random_sample((batch_size,))) < ratio_log

        final_mask = np.logical_and(mask, mask2)
        # update self.config
        self.config[final_mask] = flip_config[final_mask]

        acceptance_ratio = np.sum(final_mask)/batch_size
        return acceptance_ratio

    def local_E_Ising_batch(self, config_arr):
        '''
        To compute the Energz of 2d Transverse Field Ising model with
        the configuration given in config_array.

        H_TFI = -J*SzSz -g*Sx -h*Sz

        Basic idea is due to the fact that
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du

        So we compute (SzSz-0.5) * 2 * J / 4

        For the Sx term, we compute the amp for config_array_flip
        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=float or complex
                dtype depends on whether we are using complex amplitude wavefunction.
        '''
        J=1.
        g=2.
        h=0.
        PBC = self.PBC

        num_config, Lx, Ly, local_dim = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        # S_ij dot S_(i+1)j
        # PBC
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * (-J)
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * (-J)

        ########################
        # PBC : S_ij dot S_i(j+1)
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * (-J)
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * (-J)

        #################
        #  g sigma^x_i  #
        #################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += g (flip_Amp) / oldAmp
        localE_arr += np.einsum('ijk -> k',  flip_Amp_arr) / oldAmp * (-g)

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        Sz = (config_arr[:, :, :, 0] - 0.5) * 2
        localE_arr += np.einsum('ijk->i', -Sz * h)

        return localE_arr

    def local_E_Ising_batch_log(self, config_arr):
        '''
        See explanation in local_E_Ising_batch.
        The difference here is we compute exp(log_amp1-log_amp2),
        instead of amp1/amp2, to improve numerical stability.

        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=complex
        '''
        J=1.
        g=3.5
        h=0.
        PBC = self.PBC

        num_config, Lx, Ly, local_dim = config_arr.shape
        old_log_amp = self.eval_log_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=self.NP_COMPLEX)

        # S_ij dot S_(i+1)j
        # PBC
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * (-J)
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * (-J)

        ########################
        # PBC : S_ij dot S_i(j+1)
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * (-J)
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * (-J)

        #################
        #  g sigma^x_i  #
        #################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]

        flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                           Lx, Ly, local_dim))
        flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # localE += g * amp_ratio
        amp_ratio = np.exp(flip_log_amp_arr -
                           np.einsum('jk,i->jki', np.ones((Lx,Ly), dtype=self.NP_FLOAT), old_log_amp)
                          )
        localE_arr += np.einsum('ijk -> k',  amp_ratio) * (-g)

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        Sz = (config_arr[:, :, :, 0] - 0.5) * 2
        localE_arr += np.einsum('ijk->i', -Sz * h)

        return localE_arr

    def local_E_2dAFH_batch(self, config_arr, J=1):
        '''
        To compute the Energz of 2d Heisenberg model with
        the configuration given in config_array.

        Basic idea is due to the fact that
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        (2017version)

        So we compute (SzSz-0.5) * 2 * J / 4
        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=float or complex
                dtype depends on whether we are using complex amplitude wavefunction.
        '''
        PBC = self.PBC
        num_config, Lx, Ly, local_dim = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)

        # S_ij dot S_(i+1)j
        # PBC
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz)[:,:-1,:], flip_Amp_arr[:-1,:,:]) * J / oldAmp / 2

        ########################
        # PBC : S_ij dot S_i(j+1)
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz)[:,:,:-1], flip_Amp_arr[:,:-1,:]) * J / oldAmp / 2

        return localE_arr

    def local_E_2dAFH_batch_log(self, config_arr, J=1):
        '''
        To compute the Energy of 2d Heisenberg model with
        the configuration given in config_array.

        See explanation in local_E_2dAFH_batch.
        The difference here is we compute exp(log_amp1-log_amp2),
        instead of amp1/amp2, to improve numerical stability.

        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=complex
                regardless of using real/complex amplitude wavefunction.

        To improve numerical stability:
            1.) We evaluate only the log of the amp
            2.) We evaluate only the dataset that is relevant.
            There is an unknown error that if there is -np.inf in log_amp for some data point.
            The evaluation of that data point with the whole dataset, will result in a return
            value of nan at that point.

        '''
        PBC = self.PBC
        num_config, Lx, Ly, local_dim = config_arr.shape
        old_log_amp = self.eval_log_amp_array(config_arr)
        # old_log_amp shape (num_config,1)
        localE_arr = np.zeros((num_config), dtype=self.NP_COMPLEX)

        # S_ij dot S_(i+1)j
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]


        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:-1,:], amp_ratio[:,:-1,:]) * J / 2

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # flip_log_amp_arr = np.where((1-SzSz).transpose([1,2,0])==1, flip_log_amp_arr, np.zeros_like(flip_log_amp_arr))
        # # localE += (1-SzSz).dot(amp_ratio) * J / 2
        # amp_ratio = np.exp(flip_log_amp_arr -
        #                    np.einsum('jk,i->jki', np.ones((Lx,Ly), dtype=self.NP_FLOAT), old_log_amp)
        #                   )
        # if PBC:
        #     localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J / 2
        # else:
        #     localE_arr += np.einsum('ijk,jki->i', (1-SzSz)[:,:-1,:], amp_ratio[:-1,:,:]) * J / 2


        ########################
        # The term : S_ij dot S_i(j+1)
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * J / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]

        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape: [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:,:-1], amp_ratio[:,:,:-1]) * J / 2

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # flip_log_amp_arr = np.where((1-SzSz).transpose([1,2,0])==1, flip_log_amp_arr, np.zeros_like(flip_log_amp_arr))
        # # localE += (1-SzSz).dot(amp_ratio) * J / 2
        # amp_ratio = np.exp(flip_log_amp_arr -
        #                    np.einsum('jk,i->jki', np.ones((Lx,Ly), dtype=self.NP_FLOAT), old_log_amp)
        #                   )
        # if PBC:
        #     localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J / 2
        # else:
        #     localE_arr += np.einsum('ijk,jki->i', (1-SzSz)[:,:,:-1], amp_ratio[:,:-1,:]) * J / 2


        # if np.isnan(localE_arr).any():
        #     import pdb;pdb.set_trace()

        assert( not np.isnan(localE_arr).any() )

        return localE_arr

    def local_E_2dJ1J2_batch(self, config_arr):
        '''
        To compute the Energz of 2d J1J2 model with
        the configuration given in config_array.

        Basic idea is due to the fact that
        Sz Sz Interaction
        SzSz = 1 if uu or dd
        SzSz = 0 if ud or du
        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=float or complex
                dtype depends on whether we are using complex amplitude wavefunction.
        '''
        J1 = 1.
        J2 = self.J2
        PBC = self.PBC

        num_config, Lx, Ly, local_dim = config_arr.shape
        oldAmp = self.eval_amp_array(config_arr)
        localE_arr = np.zeros((num_config), dtype=oldAmp.dtype)


        ########################
        # PBC : J1 S_ij dot S_(i+1)j
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * J1 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J1 / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J1 / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz)[:,:-1,:], flip_Amp_arr[:-1,:,:]) * J1 / oldAmp / 2


        ########################
        # PBC : J1 S_ij dot S_i(j+1)
        ########################
        # config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * J1 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J1 / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J1 / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz[:,:,:-1]), flip_Amp_arr[:,:-1,:]) * J1 / oldAmp / 2

        ########################
        # PBC : J2 S_ij dot S_(i+1)(j+1)
        ########################
        # moving the origin config 1 down and 1 left
        config_shift_copy[:, :-1, :-1, :] = config_arr[:, 1:, 1:, :]
        config_shift_copy[:, :-1, -1, :] = config_arr[:, 1:, 0, :]
        config_shift_copy[:, -1, :-1, :] = config_arr[:, 0, 1:, :]
        config_shift_copy[:, -1, -1, :] = config_arr[:, 0, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:-1] - 0.5) * 2 * J2 / 4


        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J2 / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J2 / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz[:,:-1,:-1]),
                                    flip_Amp_arr[:-1,:-1,:]) * J2 / oldAmp / 2

        ########################
        # PBC : J2 S_ij dot S_(i+1)(j-1)
        ########################
        # moving the origin config 1 up and 1 left
        config_shift_copy[:, :-1, 1:, :] = config_arr[:, 1:, :-1, :]
        config_shift_copy[:, -1, 1:, :] = config_arr[:, 0, :-1, :]
        config_shift_copy[:, :-1, 0, :] = config_arr[:, 1:, -1, :]
        config_shift_copy[:, -1, 0, :] = config_arr[:, 0, -1, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,1:] - 0.5) * 2 * J2 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :]

        flip_Amp_arr = self.eval_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
                                                                   Lx, Ly, local_dim))
        flip_Amp_arr = flip_Amp_arr.reshape((Lx, Ly, num_config))
        # localE += (1-SzSz).dot(flip_Amp) * J2 / oldAmp / 2
        if PBC:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz), flip_Amp_arr) * J2 / oldAmp / 2
        else:
            localE_arr += np.einsum('ijk,jki->i', (1-SzSz[:,:-1,1:]), flip_Amp_arr[:-1,1:,:]) * J2 / oldAmp / 2

        return localE_arr

    def local_E_2dJ1J2_batch_log(self, config_arr):
        '''
        See explanation in local_E_2dJ1J2_batch.
        The difference here is we compute exp(log_amp1-log_amp2),
        instead of amp1/amp2, to improve numerical stability.
        # (2018/11)

        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=complex
                regardless of using real/complex amplitude wavefunction.

        To improve numerical stability further:
            1.) We evaluate ONLY the log of amp
            2.) We evaluate only the dataset that is relevant
            There is an unknown error that if there is -np.inf in log_amp for some data point.
            The evaluation of that data point with the whole dataset, will result in a return
            value of nan at that point.
        # (2019/09)
        '''
        J1 = 1.
        J2 = self.J2
        PBC = self.PBC

        num_config, Lx, Ly, local_dim = config_arr.shape
        old_log_amp = self.eval_log_amp_array(config_arr)
        # old_log_amp shape (num_config,1)
        localE_arr = np.zeros((num_config), dtype=np.complex64)


        ########################
        # PBC : J1 S_ij dot S_(i+1)j
        ########################
        config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :-1, :, :] = config_arr[:, 1:, :, :]
        config_shift_copy[:, -1, :, :] = config_arr[:, 0, :, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:] - 0.5) * 2 * J1 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, j, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, j, :]

        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])

        flip_log_amp_arr = flip_log_amp_arr.reshape((num_config, Lx, Ly))
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # # localE += (1-SzSz).dot(amp_ratio) * J1 / 2
        # amp_ratio = np.exp(flip_log_amp_arr - np.einsum('jk,i->jki',
        #                                                 np.ones((Lx,Ly), dtype=np.float32),
        #                                                 old_log_amp))
        # localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J1 / 2

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J1 / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:-1,:], amp_ratio[:,:-1,:]) * J1 / 2


        ########################
        # PBC : J1 S_ij dot S_i(j+1)
        ########################
        # config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
        config_shift_copy[:, :, :-1, :] = config_arr[:, :, 1:, :]
        config_shift_copy[:, :, -1, :] = config_arr[:, :, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J1 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:,:-1] - 0.5) * 2 * J1 / 4


        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, i, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, i, (j+1) % Ly, :]


        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])

        flip_log_amp_arr = flip_log_amp_arr.reshape((num_config, Lx, Ly))
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # # localE += (1-SzSz).dot(amp_ratio) * J1 / 2
        # amp_ratio = np.exp(flip_log_amp_arr - np.einsum('jk,i->jki',
        #                                                 np.ones((Lx,Ly), dtype=np.float32),
        #                                                 old_log_amp))
        # localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J1 / 2

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J1 / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:,:-1], amp_ratio[:,:,:-1]) * J1 / 2



        ########################
        # PBC : J2 S_ij dot S_(i+1)(j+1)
        ########################
        # moving the origin config 1 down and 1 left
        config_shift_copy[:, :-1, :-1, :] = config_arr[:, 1:, 1:, :]
        config_shift_copy[:, :-1, -1, :] = config_arr[:, 1:, 0, :]
        config_shift_copy[:, -1, :-1, :] = config_arr[:, 0, 1:, :]
        config_shift_copy[:, -1, -1, :] = config_arr[:, 0, 0, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,:-1] - 0.5) * 2 * J2 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j+1) % Ly, :]

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # # localE += (1-SzSz).dot(amp_ratio) * J2 / 2
        # amp_ratio = np.exp(flip_log_amp_arr - np.einsum('jk,i->jki',
        #                                                 np.ones((Lx,Ly), dtype=np.float32),
        #                                                 old_log_amp))
        # localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J2 / 2

        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])

        flip_log_amp_arr = flip_log_amp_arr.reshape((num_config, Lx, Ly))
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J2 / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:-1,:-1], amp_ratio[:,:-1,:-1]) * J2 / 2


        ########################
        # PBC : J2 S_ij dot S_(i+1)(j-1)
        ########################
        # moving the origin config 1 up and 1 left
        config_shift_copy[:, :-1, 1:, :] = config_arr[:, 1:, :-1, :]
        config_shift_copy[:, -1, 1:, :] = config_arr[:, 0, :-1, :]
        config_shift_copy[:, :-1, 0, :] = config_arr[:, 1:, -1, :]
        config_shift_copy[:, -1, 0, :] = config_arr[:, 0, -1, :]

        #  i            j    k,  l
        # num_config , Lx , Ly, local_dim
        SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
        if PBC:
            localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J2 / 4
        else:
            localE_arr += np.einsum('ijk->i', SzSz[:,:-1,1:] - 0.5) * 2 * J2 / 4

        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        for i in range(Lx):
            for j in range(Ly):
                config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+1) % Lx, (j-1+Ly) % Ly, :]

        # flip_log_amp_arr = self.eval_log_amp_array(config_flip_arr.reshape(Lx * Ly * num_config,
        #                                                                    Lx, Ly, local_dim))
        # flip_log_amp_arr = flip_log_amp_arr.reshape((Lx, Ly, num_config))
        # # localE += (1-SzSz).dot(amp_ratio) * J2 / 2
        # amp_ratio = np.exp(flip_log_amp_arr - np.einsum('jk,i->jki',
        #                                                 np.ones((Lx,Ly), dtype=np.float32),
        #                                                 old_log_amp))
        # localE_arr += np.einsum('ijk,jki->i', (1-SzSz), amp_ratio) * J2 / 2

        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])

        flip_log_amp_arr = flip_log_amp_arr.reshape((num_config, Lx, Ly))
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J2 / 2
        else:
            localE_arr += np.einsum('ijk,ijk->i', (1-SzSz)[:,:-1,1:], amp_ratio[:,:-1,1:]) * J2 / 2

        assert( not np.isnan(localE_arr).any() )

        return localE_arr


    def local_E_2dJulian_batch_log(self, config_arr, t1=-0.1,
                                   t2=-0.9, U=32.):
        '''
        To compute the Energy of 2d Julian model with
        the configuration given in config_array.

        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=complex
                regardless of using real/complex amplitude wavefunction.
        '''
        PBC = self.PBC

        num_config, Lx, Ly, local_dim = config_arr.shape
        old_log_amp = self.eval_log_amp_array(config_arr)
        # old_log_amp shape (num_config,1)
        localE_arr = np.zeros((num_config), dtype=self.NP_COMPLEX)
        # # old--> new    index   coeff
        # # (1 --> 0)     3       1
        # # (1 --> 1)     4       sqrt(2)
        # # (2 --> 0)     6       sqrt(2)
        # # (2 --> 1)     7       2
        inter_coeff_map = np.zeros((9,))
        inter_coeff_map[3] = 1
        inter_coeff_map[4] = np.sqrt(2)
        inter_coeff_map[6] = np.sqrt(2)
        inter_coeff_map[7] = 2
        inter_coeff_map = inter_coeff_map.reshape([3,3])
        # inter_coeff_map = np.zeros((4,))
        # inter_coeff_map[2] = 1.
        # inter_coeff_map = inter_coeff_map.reshape([2, 2])


        max_num_p = 2
        #######################
        # HOP TO THE RIGHT   ##
        #######################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        interacting_coeff = np.zeros((Lx, Ly, num_config))
        for i in range(Lx):
            for j in range(Ly):
                ## hopping from (i,j) to (i+1, j)
                new_i = (i+1)%Lx
                mask1 = (config_arr[:, i, j, 0] != 1)  # (i,j) does not have zero particle
                mask2 = (config_arr[:, new_i, j, max_num_p] != 1)  # (i+1,j) does not have two particles
                final_mask = np.logical_and(mask1, mask2)

                old_site_idx = np.nonzero(config_arr[:, i, j, :])[1]
                new_site_idx = np.nonzero(config_arr[:, new_i, j, :])[1]

                # change the config
                tmp_config = config_flip_arr[i, j, final_mask, i, j, 1:].copy()
                config_flip_arr[i, j, final_mask, i, j, 1:] *= 0
                config_flip_arr[i, j, final_mask, i, j, :max_num_p] = tmp_config
                tmp_config = config_flip_arr[i, j, final_mask, new_i, j, :max_num_p].copy()
                config_flip_arr[i, j, final_mask, new_i, j, :max_num_p] *= 0
                config_flip_arr[i, j, final_mask, new_i, j, 1:] = tmp_config
                # write in the interacting_coeff
                if i%2 == 0:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t1
                else:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t2



        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, num_config, Lx, Ly]
        interacting_coeff = np.transpose(interacting_coeff, [2,0,1])
        # now of the shape, [num_config, Lx, Ly]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        # this mask is identical to the final_mask in the loop above.
        mask_to_eval = np.isclose(interacting_coeff.flatten(), np.zeros_like(interacting_coeff.flatten()))
        mask_to_eval = np.logical_not(mask_to_eval)
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        ## [TODO] remove the exp( masked ) part.
        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        # amp_ratio = np.exp(flip_log_amp_arr.reshape(num_config, Lx, Ly) -
        #                    np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT))
        #                   )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff, amp_ratio)
        else:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff[:,:-1,:], amp_ratio[:,:-1,:])


        #######################
        # HOP TO THE LEFT    ##
        #######################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        interacting_coeff = np.zeros((Lx, Ly, num_config))
        for i in range(Lx):
            for j in range(Ly):
                ## hopping from (i,j) to (i-1, j)
                new_i = (i-1+Lx)%Lx
                mask1 = (config_arr[:, i, j, 0] != 1)  # (i,j) does not have zero particle
                mask2 = (config_arr[:, new_i, j, max_num_p] != 1)  # (i-1,j) does not have two particles
                final_mask = np.logical_and(mask1, mask2)

                old_site_idx = np.nonzero(config_arr[:, i, j, :])[1]
                new_site_idx = np.nonzero(config_arr[:, new_i, j, :])[1]

                # change the config
                tmp_config = config_flip_arr[i, j, final_mask, i, j, 1:].copy()
                config_flip_arr[i, j, final_mask, i, j, 1:] *= 0
                config_flip_arr[i, j, final_mask, i, j, :max_num_p] = tmp_config
                tmp_config = config_flip_arr[i, j, final_mask, new_i, j, :max_num_p].copy()
                config_flip_arr[i, j, final_mask, new_i, j, :max_num_p] *= 0
                config_flip_arr[i, j, final_mask, new_i, j, 1:] = tmp_config
                # write in the interacting_coeff
                if i%2 == 0:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t2
                else:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t1



        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, num_config, Lx, Ly]
        interacting_coeff = np.transpose(interacting_coeff, [2,0,1])
        # now of the shape, [num_config, Lx, Ly]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        # this mask is identical to the final_mask in the loop above.
        mask_to_eval = np.isclose(interacting_coeff.flatten(), np.zeros_like(interacting_coeff.flatten()))
        mask_to_eval = np.logical_not(mask_to_eval)
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        ## [TODO] remove the exp( masked ) part.
        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        # amp_ratio = np.exp(flip_log_amp_arr.reshape(num_config, Lx, Ly) -
        #                    np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT))
        #                   )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff, amp_ratio)
        else:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff[:,1:,:], amp_ratio[:,1:,:])


        #####################
        # HOP TO THE UP   ##
        #####################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        interacting_coeff = np.zeros((Lx, Ly, num_config))
        for i in range(Lx):
            for j in range(Ly):
                ## hopping from (i,j) to (i, j+1)
                new_j = (j+1)%Ly
                mask1 = (config_arr[:, i, j, 0] != 1)  # (i,j) does not have zero particle
                mask2 = (config_arr[:, i, new_j, max_num_p] != 1)  # (i,j+1) does not have two particles
                final_mask = np.logical_and(mask1, mask2)

                old_site_idx = np.nonzero(config_arr[:, i, j, :])[1]
                new_site_idx = np.nonzero(config_arr[:, i, new_j, :])[1]

                # change the config
                tmp_config = config_flip_arr[i, j, final_mask, i, j, 1:].copy()
                config_flip_arr[i, j, final_mask, i, j, 1:] *= 0
                config_flip_arr[i, j, final_mask, i, j, :max_num_p] = tmp_config
                tmp_config = config_flip_arr[i, j, final_mask, i, new_j, :max_num_p].copy()
                config_flip_arr[i, j, final_mask, i, new_j, :max_num_p] *= 0
                config_flip_arr[i, j, final_mask, i, new_j, 1:] = tmp_config
                # write in the interacting_coeff
                if j%2 == 0:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t1
                else:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t2


        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, num_config, Lx, Ly]
        interacting_coeff = np.transpose(interacting_coeff, [2,0,1])
        # now of the shape, [num_config, Lx, Ly]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        # this mask is identical to the final_mask in the loop above.
        mask_to_eval = np.isclose(interacting_coeff.flatten(), np.zeros_like(interacting_coeff.flatten()))
        mask_to_eval = np.logical_not(mask_to_eval)
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        ## [TODO] remove the exp( masked ) part.
        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        # amp_ratio = np.exp(flip_log_amp_arr.reshape(num_config, Lx, Ly) -
        #                    np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT))
        #                   )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff, amp_ratio)
        else:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff[:,:,:-1], amp_ratio[:,:,:-1])


        #######################
        # HOP TO THE DOWN    ##
        #######################
        #   g      h      i           j    k     l
        #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin
        config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
        interacting_coeff = np.zeros((Lx, Ly, num_config))
        for i in range(Lx):
            for j in range(Ly):
                ## hopping from (i,j) to (i, j-1)
                new_j = (j-1+Ly)%Ly
                mask1 = (config_arr[:, i, j, 0] != 1)  # (i,j) does not have zero particle
                mask2 = (config_arr[:, i, new_j, max_num_p] != 1)  # (i,j-1) does not have two particles
                final_mask = np.logical_and(mask1, mask2)

                old_site_idx = np.nonzero(config_arr[:, i, j, :])[1]
                new_site_idx = np.nonzero(config_arr[:, i, new_j, :])[1]

                # change the config
                tmp_config = config_flip_arr[i, j, final_mask, i, j, 1:].copy()
                config_flip_arr[i, j, final_mask, i, j, 1:] *= 0
                config_flip_arr[i, j, final_mask, i, j, :max_num_p] = tmp_config
                tmp_config = config_flip_arr[i, j, final_mask, i, new_j, :max_num_p].copy()
                config_flip_arr[i, j, final_mask, i, new_j, :max_num_p] *= 0
                config_flip_arr[i, j, final_mask, i, new_j, 1:] = tmp_config
                # write in the interacting_coeff
                if j%2 == 0:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t2
                else:
                    interacting_coeff[i, j, :] = inter_coeff_map[old_site_idx, new_site_idx] * t1



        config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
        # now of the shape, [num_config, Lx, Ly, num_config, Lx, Ly]
        interacting_coeff = np.transpose(interacting_coeff, [2,0,1])
        # now of the shape, [num_config, Lx, Ly]
        flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
        # this mask is identical to the final_mask in the loop above.
        mask_to_eval = np.isclose(interacting_coeff.flatten(), np.zeros_like(interacting_coeff.flatten()))
        mask_to_eval = np.logical_not(mask_to_eval)
        flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

        ## [TODO] remove the exp( masked ) part.
        amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
        mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
        amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                          np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                        )
        # amp_ratio = np.exp(flip_log_amp_arr.reshape(num_config, Lx, Ly) -
        #                    np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT))
        #                   )

        if PBC:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff, amp_ratio)
        else:
            localE_arr += np.einsum('ijk,ijk->i', interacting_coeff[:,:,1:], amp_ratio[:,:,1:])

        #################
        #  h sigma^z_i  #
        #################
        # num_config x L
        localE_arr += np.einsum('ijk->i', config_arr[:, :, :, max_num_p] * U)

        ###################################
        # mu if N not equal to Lx*Ly//2  #
        ###################################
        mu = 0.
        num_particle = np.sum(config_arr, axis=(1,2)).dot(np.arange(max_num_p+1))
        # localE_arr += mu * (num_particle != Lx*Ly//2)
        # localE_arr += mu * num_particle
        print("num_batch in LxLy//2 sector : ", np.sum(num_particle == Lx*Ly//2)/num_config)

        if np.isnan(localE_arr).any():
            import pdb;pdb.set_trace()

        return localE_arr, localE_arr
        # return localE_arr + 0.5 * (num_particle - Lx*Ly//2)**2, localE_arr

    def local_totalS_batch_log(self, config_arr, J=1):
        '''
        To compute the expectation value of the total spin operator
        \sum_alpha  <  \sum_i (Si^alpha) \sum_j (Sj^alpha) >

        See explanation in local_E_2dAFH_batch.
        The difference here is we compute exp(log_amp1-log_amp2),
        instead of amp1/amp2, to improve numerical stability.

        Input:
            config_arr:
                np.array of shape (num_config, Lx, Ly, local_dim)
                dtype=np.int
        Output:
            localE_arr:
                np.array of shape (num_config)
                dtype=complex
                regardless of using real/complex amplitude wavefunction.

        '''
        num_config, Lx, Ly, local_dim = config_arr.shape
        old_log_amp = self.eval_log_amp_array(config_arr)
        # old_log_amp shape (num_config,1)
        localE_arr = np.zeros((num_config), dtype=self.NP_COMPLEX)

        for a in range(Lx):
            for b in range(Ly):
                if a == 0 and b == 0:
                    localE_arr += 3./4 * Lx * Ly
                    continue
                elif a == 0:
                    # S_ij dot S_(i+a)(j+b)
                    config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
                    config_shift_copy[:, :, :-b, :] = config_arr[:, :, b:, :]
                    config_shift_copy[:, :, -b:, :] = config_arr[:, :, :b, :]
                elif b == 0:
                    # S_ij dot S_(i+a)(j+b)
                    config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
                    config_shift_copy[:, :-a, :, :] = config_arr[:, a:, :, :]
                    config_shift_copy[:, -a:, :, :] = config_arr[:, :a, :, :]
                else:
                    # S_ij dot S_(i+a)(j+b)
                    config_shift_copy = np.zeros((num_config, Lx, Ly, local_dim), dtype=np.int32)
                    config_shift_copy[:, :-a, :-b, :] = config_arr[:, a:, b:, :]
                    config_shift_copy[:, :-a, -b:, :] = config_arr[:, a:, :b, :]
                    config_shift_copy[:, -a:, :-b, :] = config_arr[:, :a, b:, :]
                    config_shift_copy[:, -a:, -b:, :] = config_arr[:, :a, :b, :]


                #  i            j    k,  l
                # num_config , Lx , Ly, local_dim
                SzSz = np.einsum('ijkl,ijkl->ijk', config_arr, config_shift_copy)
                ## Always PBC:
                localE_arr += np.einsum('ijk->i', SzSz - 0.5) * 2 * J / 4

                #   g      h      i           j    k     l
                #   Lx ,  Ly , num_config ,  Lx , Ly,  num_spin=local_dim
                config_flip_arr = np.einsum('gh,ijkl->ghijkl', np.ones((Lx, Ly), dtype=np.int32), config_arr)
                for i in range(Lx):
                    for j in range(Ly):
                        config_flip_arr[i, j, :, i, j, :] = 1 - config_flip_arr[i, j, :, i, j, :]
                        config_flip_arr[i, j, :, (i+a) % Lx, (j+b) % Ly, :] = 1 - config_flip_arr[i, j, :, (i+a) % Lx, (j+b) % Ly, :]


                config_flip_arr = np.transpose(config_flip_arr, [2,0,1,3,4,5])
                # now of the shape, [num_config, Lx, Ly, Lx, Ly, local_dim]
                flip_log_amp_arr = np.zeros([num_config*Lx*Ly],dtype=self.NP_COMPLEX)
                mask_to_eval = np.isclose(SzSz.flatten(), np.zeros_like(SzSz.flatten()))  # mask part indicate Szi != Szj
                flip_log_amp_arr[mask_to_eval] = self.eval_log_amp_array(config_flip_arr.reshape(num_config*Lx*Ly, Lx, Ly, local_dim)[mask_to_eval])

                amp_ratio = np.zeros([num_config, Lx, Ly], dtype=self.NP_COMPLEX)
                mask_to_eval = mask_to_eval.reshape([num_config, Lx, Ly])
                amp_ratio[mask_to_eval] = np.exp((flip_log_amp_arr.reshape(num_config, Lx, Ly) -
                                                  np.einsum('i,jk->ijk', old_log_amp, np.ones((Lx,Ly), dtype=self.NP_FLOAT)))[mask_to_eval]
                                                )

                # Always PBC:
                localE_arr += np.einsum('ijk,ijk->i', (1-SzSz), amp_ratio) * J / 2

        assert( not np.isnan(localE_arr).any() )

        return localE_arr


############################
#  END OF DEFINITION NQS2d #
############################
