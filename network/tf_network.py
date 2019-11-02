import numpy as np
import math
import tensorflow as tf
# from .hoshen_kopelman import  label
from . import tf_wrapper as tf_
from . import mask

# The following argument is for setting up kfac optimizer
import sys
# We append the relative path to kfac cloned directory
sys.path.append("../../kfac")
import kfac
from kfac.python.ops.loss_functions import LogProbLoss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


class tf_network:
    def __init__(self, which_net, inputShape, optimizer, dim, sess=None,
                 learning_rate=0.1125, momentum=0.90, alpha=2,
                 activation=None, using_complex=True, single_precision=True,
                 batch_size=None, using_symm=False, num_blocks=10, multi_gpus=False):
        '''
        Arguments as follows:
        which_net:
            the name of the network structure choosen.
        inputShape:
            x.shape
        dim:
            dimension of the physical system
        using_complex:
            using complex-valued/real-valued wavefunction or not.
            This would affect the type of other variables, including
            local energy.
        '''
        if single_precision:
            self.TF_FLOAT = tf.float32
            self.NP_FLOAT = np.float32
            self.TF_COMPLEX = tf.complex64
            self.NP_COMPLEX = np.complex64
        else:
            self.TF_FLOAT = tf.float64
            self.NP_FLOAT = np.float64
            self.TF_COMPLEX = tf.complex128
            self.NP_COMPLEX = np.complex128
        self.TF_INT = tf.int8
        ##################
        # Parameters
        ##################
        self.learning_rate = tf.Variable(learning_rate, name='learning_rate')
        self.momentum = tf.Variable(momentum, name='momentum')
        self.exp_stabilizer = tf.Variable(0., name="exp_stabilizer", dtype=self.TF_FLOAT)
        self.global_step = tf.Variable(0, name="global_step")
        # dropout = 0.75  # Dropout, probability to keep units
        self.bn_is_training = tf.placeholder(tf.bool)
        self.max_bp_batch_size = 512
        ## [TODO]
        ## We put no restriction in the forward batchsize here,
        ## The restriction is put in the quantum state NQS evaluation function.
        ## To not exceed the size.
        # self.max_fp_batch_size = self.max_bp_batch_size * 10
        self.batch_size = batch_size


        ##########################
        # tf Graph input & Create network
        ##########################
        self.alpha = alpha
        self.activation = activation
        self.which_net = which_net
        self.num_blocks = num_blocks
        self.using_complex = using_complex
        self.using_symm = using_symm
        self.keep_prob = tf.placeholder(self.TF_FLOAT)
        self.dx_exp_stabilizer = tf.placeholder(self.TF_FLOAT)
        self.multi_gpus = multi_gpus
        # Define
        # E_loc_m_avg  <-- E_loc minus avg(E_loc)
        # which is useful for plain gradient descent algorithm
        if using_complex:
            self.E_loc_m_avg = tf.placeholder(self.TF_COMPLEX, [None, 1])
            # complex-valued wavefunction will result in complex local Energy
        else:
            self.E_loc_m_avg = tf.placeholder(self.TF_FLOAT, [None, 1])
            # real-valued wavefunction will result in real local Energy

        if dim == 1:
            self.x = tf.placeholder(self.TF_INT, [None, inputShape[0], inputShape[1]])
            self.L = int(inputShape[0])
            self.build_network = self.build_network_1d
        elif dim == 2:
            if which_net in ['pre_sRBM']:
                self.x = tf.placeholder(self.TF_INT, [None, inputShape[0], inputShape[1], 4])
            else:
                self.x = tf.placeholder(self.TF_INT, [None, inputShape[0], inputShape[1], inputShape[2]])

            self.Lx = int(inputShape[0])
            self.Ly = int(inputShape[1])
            assert self.Lx == self.Ly
            self.LxLy = self.Lx * self.Ly
            self.channels = int(inputShape[2])
            if self.channels > 3:
                print("Not yet implemented for tJ, Hubbard model")
                raise NotImplementedError
            else:
                pass

            self.ordering = mask.gen_raster_scan_order(self.Lx)
            self.build_network = self.build_network_2d

        else:
            raise NotImplementedError

        # Define layer_collection
        # layer_collection is defined before the build_ntwork
        # so that one can register the layer while building the network.
        if optimizer == "KFAC":
            self.layer_collection = kfac.LayerCollection()
            self.registered = False
        else:
            self.layer_collection = None
            self.registered = False

        ### [TODO]
        ### Add for loop if multi gpu
        all_out = self.build_network(which_net, self.x, self.activation, self.num_blocks)
        if len(all_out) == 2:
            self.amp, self.log_amp = all_out[:2]
        else:
            try:
                self.amp, self.log_amp, self.log_cond_amp, self.prob = all_out[:4]
                if self.using_symm:
                    # self.symm_amp = all_out[4]
                    # self.symm_log_amp = all_out[5]
                    self.amp = all_out[4]
                    self.log_amp = all_out[5]
                    self.symm_prob = all_out[7]
                else:
                    pass

            except:
                raise Exception


        if self.log_amp is None:
            # self.log_amp = tf.log(self.amp)
            # For real-valued wavefunction, log_amp is only a intermediate step for
            # Log gradient. log_amp should not be read out. Otherwise, one need to
            # define as below,
            #
            self.log_amp = tf.log(tf.cast(self.amp, self.TF_COMPLEX))
            # Cast type to complex before log, to prevent nan in
            # the case for tf.float input < 0
            #
            # This is not a problem if one do not want to read out
            # The log value explicitly, but only need the derivative of log.
            #
            # But to prevent error, always cast to TF_COMPLEX.

        #########################
        # Variables Creation
        #########################
        self.model_var_list = tf.global_variables()
        if tf.get_default_graph().get_name_scope() == '':
            current_scope = 'network'
        else:
            current_scope = tf.get_default_graph().get_name_scope() + '/network'

        self.para_list_w_bn = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=current_scope)
        print("We now have these variables:")
        self.para_list_wo_bn=[]
        for i in self.para_list_w_bn:
            print(i.name)
            if 'bn' in i.name:
                pass
            else:
                self.para_list_wo_bn.append(i)

        print("Created list of variables without batchnorm")
        self.para_list = self.para_list_wo_bn
        # para_list are list of variables for gradient calculation
        # should not include batchnorm variables.
        #
        # Below, we further separate the variables into real and imaginary
        # part of the variables.
        # The separation is necessary for construction of wirtinger derivative.

        self.var_shape_list = [var.get_shape().as_list() for var in self.para_list]
        self.num_para = self.getNumPara()
        # Define optimizer
        if optimizer == "KFAC":
            # self.layer_collection.register_categorical_predictive_distribution(tf.square(self.amp),
            #                                                                    targets=tf.square(self.amp))
            # kfac_loss = LogProbLoss(self.prob)
            kfac_loss = LogProbLoss(self.amp)
            # kfac_loss = kfac.python.ops.loss_functions.LogProbLoss(tf.square(self.amp))
            self.layer_collection._register_loss_function(kfac_loss, self.amp, 'log_prob_loss')
            # using amp as P, to compute <O^*O> = <logp logp>
            # Error might occur for not taking complex conjugate ???
            # for var in self.para_list:
            #     self.layer_collection.define_linked_parameters(var)

            # self.layer_collection.auto_register_layers(var_list=self.para_list,
            #                                            batch_size=self.batch_size)
            self.layer_collection.auto_register_layers(var_list=self.para_list)
            # self.layer_collection.register_loss_function(-self.log_amp, self.amp, 'loss_base')
        else:
            pass

        self.optimizer = tf_.select_optimizer(optimizer, self.learning_rate,
                                              self.momentum, var_list=self.para_list,
                                              layer_collection=self.layer_collection)

        self.re_para_idx = [i for i in range(len(self.para_list)) if 're' in self.para_list[i].name]
        self.im_para_idx = [i for i in range(len(self.para_list)) if 'im' in self.para_list[i].name]
        print("self.re_para_idx : ", self.re_para_idx)
        print("self.im_para_idx : ", self.im_para_idx)
        # assert {*self.re_para_idx} | {*self.im_para_idx} == {*range(len(self.para_list))}
        # assert len( {*self.re_para_idx} & {*self.im_para_idx} ) == 0

        self.im_para_array = np.zeros(self.num_para, dtype=np.int)
        im_para_array_idx=0
        for idx, var_shape in enumerate(self.var_shape_list):
            tmp_idx = im_para_array_idx + np.prod(var_shape)
            if idx in self.re_para_idx:
                im_para_array_idx = tmp_idx
            else:
                self.im_para_array[im_para_array_idx:tmp_idx] = 1
                im_para_array_idx = tmp_idx

        if not using_complex:
            self.im_para_array = np.zeros(self.num_para, dtype=np.int)

        # Below we define the gradient.
        # tf.gradient(cost, variable_list)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Log Gradient, loss = log(wave function)
            # Define Energy Gradient, loss = E(wave function)
            if not using_complex:
                # (1.1) real log_grads
                # real-valued wavefunction, log_amp, grad_log_amp are all real
                self.log_grads = tf.gradients(self.log_amp, self.para_list, grad_ys=tf.complex(1.,0.))
                # (2.1)
                self.E_grads = tf.gradients(self.log_amp, self.para_list, grad_ys=tf.complex(self.E_loc_m_avg,0.))
                # log_amp is always complex, so we need to specify grad_ys
            else:
                # (1.2) complex log_grads
                # complex-valued wavefunction, log_amp are all complex, but
                # grad_log_amp would be cast to real by tensorflow default !
                # To prevent this, we manually compute the gradient.
                # Cast gradient back to real only before apply_gradient.
                #
                # Computing O not O^* here.
                # We explicitly do the confugation with numpy array.
                self.log_grads_real = tf.gradients(tf.real(self.log_amp), self.para_list)
                self.log_grads_imag = tf.gradients(tf.imag(self.log_amp), self.para_list)
                ##############
                # Method 1. ##
                ##############
                # We simply combine the grad of the real part and the imag part.
                ##############
                self.log_grads = [tf.complex(self.log_grads_real[i], self.log_grads_imag[i])
                                  for i in range(len(self.log_grads_real))]
                ##############
                # Method 2. ##
                ##############
                # We reorder the gradient derivative by the definition of wirtinger derivative.
                # Weirdly this does not give the correct value gradient value after calculation
                # and comparing to plain gradient desecent.
                ##############
                # self.log_grads = [None]*len(self.log_grads_real)
                # for idx in range(len(self.re_para_idx)):
                #     re_idx = self.re_para_idx[idx]
                #     im_idx = self.im_para_idx[idx]
                #     dre_re = self.log_grads_real[re_idx]
                #     dre_im = self.log_grads_imag[re_idx]
                #     dim_re = self.log_grads_real[im_idx]
                #     dim_im = self.log_grads_imag[im_idx]
                #     self.log_grads[re_idx] = (dre_re) / 2.
                #     self.log_grads[im_idx] = (dim_re) / 2.
                #     # self.log_grads[re_idx] = (dre_re + dim_im) / 2.
                #     # O^*
                #     # self.log_grads[im_idx] = (dim_re - dre_im) / 2.
                #     # O
                #     # self.log_grads[im_idx] = (-dim_re + dre_im) / 2.

                # (2.1)
                self.E_grads = tf.gradients(self.log_amp, self.para_list, grad_ys=self.E_loc_m_avg)
                # log_amp is always complex, so we need to specify grad_ys


        # Pseudo Code for batch Gradient
        # Method 1: Copying network
        #
        # examples = tf.split(self.x)
        # weight_copies = [tf.identity(self.para_list) for x in examples]
        # output = tf.stack(f(x, w) for x, w in zip(examples, weight_copies))
        # cost = tf.log(output)
        # per_example_gradients = tf.gradients(cost, weight_copies)

        # Method 2: Unaggregated Gradient with while_loop
        self.unaggregated_gradient = self.build_unaggregated_gradient()
        self.variance_log_gradient = self.build_variance_log_gradient()

        # Do some operation on grads.
        # We apply CG/MINRES to obtain natural gradient.
        # Get the new gradient from outside by placeholder
        self.newgrads = [tf.placeholder(self.TF_FLOAT, g.get_shape()) for g in self.log_grads]
        ### SINCE THE NEW GRAD should be coming from the grad of the energy minimization
        ### It should be a real value gradient
        ### Since we now consider 
        if self.using_complex:
            self.newgrads_kfac = [tf.placeholder(self.TF_FLOAT, g.get_shape()) for g in self.log_grads]
            # self.newgrads_kfac = [tf.placeholder(self.TF_COMPLEX, g.get_shape()) for g in self.log_grads]
        else:
            self.newgrads_kfac = [tf.placeholder(self.TF_FLOAT, g.get_shape()) for g in self.log_grads]


        if optimizer == "KFAC":
            self.Momoptimizer = tf_.select_optimizer('Mom', self.learning_rate,
                                                     self.momentum, var_list=self.para_list,
                                                     layer_collection=None)
            self.train_op = self.Momoptimizer.apply_gradients(list(zip(self.newgrads,
                                                                       self.para_list)))
        else:
            self.train_op = self.optimizer.apply_gradients(list(zip(self.newgrads,
                                                                    self.para_list)))

        self.update_exp_stabilizer = self.exp_stabilizer.assign(self.exp_stabilizer +
                                                                self.dx_exp_stabilizer)

        if optimizer == "KFAC":
            cov_update_thunks, inv_update_thunks = self.optimizer.make_vars_and_create_op_thunks()
            def make_update_op(update_thunks):
                update_ops = [thunk() for thunk in update_thunks]
                return tf.group(*update_ops)

            cov_update_op = make_update_op(cov_update_thunks)
            # with tf.control_dependencies([cov_update_op]):
            inverse_op=tf.cond(tf.equal(tf.mod(self.global_step, 10), 0),
                               lambda: make_update_op(inv_update_thunks), tf.no_op)
            #     with tf.control_dependencies([inverse_op]):
            self.cov_update_op = cov_update_op
            self.inverse_op = inverse_op

            self.fisher_inverse = self.optimizer._fisher_est.multiply_inverse(list(zip(self.newgrads_kfac,
                                                                                       self.para_list)))
            self.fisher_multiply = self.optimizer._fisher_est.multiply(list(zip(self.newgrads_kfac,
                                                                                self.para_list)))



        ### 
        # 1. get_amp; get_log_amp functions are done with forward pass of NN
        # 2. get_log_grads; get_E_grads functions are done with back propagation
        #
        # The "plain" refer to no prerpocessing
        # The "pre" refer to preprocessing the input data with augmentation
        ### 
        if which_net in ['pre_sRBM']:
            self.get_amp = self.pre_get_amp
            self.get_log_grads = self.pre_get_log_grads
            self.get_E_grads = self.pre_get_E_grads
        else:
            self.get_amp = self.plain_get_amp
            self.get_log_amp = self.plain_get_log_amp
            self.get_log_grads = self.plain_get_log_grads
            self.get_E_grads = self.plain_get_E_grads


        ####################################################################################
        # Initializing All the variables and operation, all operation and variables should 
        # be defined before here.!!!!
        ####################################################################################
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        if sess is None:
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess

    def run_global_variables_initializer(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def enrich_features(self, X0):
        '''
        python based method to precoss the input array
        to add more input features
        '''
        X0_shape = X0.shape
        pos_label = label(X0[:,:,:,0]).reshape(X0_shape[:-1]+(1,))
        neg_label = label(X0[:,:,:,1]).reshape(X0_shape[:-1]+(1,))
        new_X = np.concatenate([X0, pos_label, neg_label], axis=-1)
        return new_X

    def plain_get_amp(self, X0):
        return self.sess.run(self.amp, feed_dict={self.x: X0, self.bn_is_training: False})

    def plain_get_log_amp(self, X0):
        return self.sess.run(self.log_amp, feed_dict={self.x: X0, self.bn_is_training: False})

    def plain_get_log_grads(self, X0):
        return self.sess.run(self.log_grads, feed_dict={self.x: X0, self.bn_is_training: False})

    def plain_get_cond_log_amp(self, X0):
        return self.sess.run(self.log_cond_amp, feed_dict={self.x: X0, self.bn_is_training: False})

    def build_unaggregated_gradient(self):
        '''
        We build the while loop operation to parallelize the per example
        gradient calculation. This also avoid the CPU-GPU transfer per example.
        The while_loop is parallelized with the default value parallel_iter=10.
        Input:
            self.x, placeholder. Inside the while loop, being splitted and with network built
            and gradient computed separately.
        Return:
            log_derivative per example in matrix with shape (num_para, num_data)

        alternative implementation:
        https://stackoverflow.com/questions/38994037/tensorflow-while-loop-for-training
        '''
        if self.using_complex:
            unaggregated_grad = tf.TensorArray(dtype=self.TF_COMPLEX, size=tf.shape(self.x)[0])
            # unaggregated_grad = tf.TensorArray(dtype=self.TF_FLOAT, size=tf.shape(self.x)[0])
        else:
            unaggregated_grad = tf.TensorArray(dtype=self.TF_FLOAT, size=tf.shape(self.x)[0])

        init_state = (0, unaggregated_grad)
        # i = tf.constant(0)
        condition = lambda i, _: i < tf.shape(self.x)[0]
        def body(i, ta):
            single_x = self.x[i:i+1]
            if self.using_complex:
                single_all_out = self.build_network(self.which_net, single_x,
                                                    self.activation, self.num_blocks)
                single_amp, single_log_amp = single_all_out[:2]
                try:
                    single_log_cond_amp, single_prob = single_all_out[2:4]
                except:
                    print(" NO NAQS USED !!! ")

                # single_amp, single_log_amp = self.build_network(self.which_net, single_x,
                #                                                  self.activation)

                if single_log_amp is None:
                    single_log_amp = tf.log(tf.cast(single_amp, self.TF_COMPLEX))

                ##############
                # Method 1
                ##############
                single_log_grads_real = tf.gradients(tf.real(single_log_amp), self.para_list)
                single_log_grads_imag = tf.gradients(tf.imag(single_log_amp), self.para_list)
                single_log_grads = [tf.complex(single_log_grads_real[j], single_log_grads_imag[j])
                                    for j in range(len(single_log_grads_real))]
                ##############
                # Method 2
                ##############
                # single_log_grads = [None]*len(single_log_grads_real)
                # for idx in range(len(self.re_para_idx)):
                #     re_idx = self.re_para_idx[idx]
                #     im_idx = self.im_para_idx[idx]
                #     single_dre_re = single_log_grads_real[re_idx]
                #     single_dre_im = single_log_grads_imag[re_idx]
                #     single_dim_re = single_log_grads_real[im_idx]
                #     single_dim_im = single_log_grads_imag[im_idx]
                #     single_log_grads[re_idx] = (single_dre_re ) / 2.
                #     single_log_grads[im_idx] = (single_dim_re) / 2.
                #     # single_log_grads[re_idx] = (single_dre_re + single_dim_im) / 2.
                #     # O^*
                #     # single_log_grads[im_idx] = (single_dim_re - single_dre_im) / 2.
                #     # O
                #     # single_log_grads[im_idx] = (-single_dim_re + single_dre_im) / 2.


                ta = ta.write(i, tf.concat([tf.reshape(g,[-1]) for g in single_log_grads], axis=0 ))
            else:
                single_all_out = self.build_network(self.which_net, single_x,
                                                    self.activation, self.num_blocks)
                single_amp, single_log_amp = single_all_out[:2]
                try:
                    single_log_cond_amp, single_prob = single_all_out[2:4]
                except:
                    print(" NO NAQS USED !!! ")

                # single_amp, single_log_amp = self.amp, self.log_amp
                if single_log_amp is None:
                    single_log_amp = tf.log(tf.cast(single_amp, self.TF_COMPLEX))

                # single_log_amp = tf.log(tf.cast(self.build_network(self.which_net, single_x, self.activation)[0], self.TF_COMPLEX))
                ta = ta.write(i, tf.concat([tf.reshape(g,[-1]) for g in tf.gradients(single_log_amp, self.para_list, grad_ys=tf.complex(1.,0.))], axis=0 ))
            return (i+1, ta)

        n, final_unaggregated_grad = tf.while_loop(condition, body, init_state, back_prop=False)
        final_unaggregated_grad = tf.transpose(final_unaggregated_grad.stack())
        return final_unaggregated_grad

    def build_variance_log_gradient(self):
        tf_O_array = self.unaggregated_gradient
        exp_sq_log_grad = tf.real(tf.reduce_mean(tf.math.conj(tf_O_array) * tf_O_array, axis=1))
        exp_log_grad = tf.reduce_mean(tf_O_array, axis=1)
        sq_exp_log_grad = tf.real(tf.math.conj(exp_log_grad) * exp_log_grad)
        return exp_sq_log_grad - sq_exp_log_grad

    def run_unaggregated_gradient(self, X0):
        return self.sess.run(self.unaggregated_gradient, feed_dict={self.x: X0, self.bn_is_training: False})

    def run_variance_log_gradient(self, X0):
        return self.sess.run(self.variance_log_gradient, feed_dict={self.x: X0, self.bn_is_training: False})

    def plain_get_E_grads(self, X0, E_loc_array):
        # Implementation below fail for unknown reason
        # Not sure whether it is bug from tensorflow or not.
        # 
        # E_vec = (self.E_loc - tf.reduce_mean(self.E_loc))
        # E = tf.reduce_sum(tf.multiply(E_vec, log_amp))
        # E = (tf.multiply(E_vec, log_amp))
        # return self.sess.run(tf.gradients(E, self.para_list),

        # because grad_ys has to have the same shape as ys
        # we need to reshape E_loc_array as [None, 1]
        '''
        Input (numpy array):
            X0 : the config array
            E_loc_array : E_array - E_avg
        Output:
            returning the gradient, in python list of numpy array.
            The gradient is always real. No matter whether the amplitudes
            are complex or real.

        Note:
        1.  max_bp_batch_size should be tuned to not exceeding the
            memory limit. The larger, the faster.
        2.  We parametrized the network with "REAL PARAMETERS",
            so the gradient should be real and is real by tensorflow
            default. The grad_array is with dtype NP_FLOAT
        '''
        E_loc_array = E_loc_array.reshape([-1, 1])
        num_data = E_loc_array.size
        max_bp_size = self.max_bp_batch_size
        if num_data > max_bp_size:
            grad_array = np.zeros((self.num_para, ), dtype=self.NP_FLOAT)
            for idx in range(num_data // max_bp_size):
                G_list = self.sess.run(self.E_grads,
                                       feed_dict={self.x: X0[max_bp_size*idx : max_bp_size*(idx+1)],
                                                  self.bn_is_training: True,
                                                  self.E_loc_m_avg: E_loc_array[max_bp_size*idx : max_bp_size*(idx+1)]})
                grad_array += np.concatenate([g.flatten() for g in G_list])

            if num_data % max_bp_size != 0:
                G_list = self.sess.run(self.E_grads,
                                       feed_dict={self.x: X0[max_bp_size*(num_data//max_bp_size):],
                                                  self.bn_is_training: True,
                                                  self.E_loc_m_avg: E_loc_array[max_bp_size*(num_data//max_bp_size):]})
            grad_array += np.concatenate([g.flatten() for g in G_list])
            G_list = []
            grad_ind = 0
            for var_shape in self.var_shape_list:
                var_size = np.prod(var_shape)
                G_list.append(grad_array[grad_ind:grad_ind + var_size].reshape(var_shape))
                grad_ind += var_size

            return G_list
        else:
            return self.sess.run(self.E_grads,
                                 feed_dict={self.x: X0,
                                            self.bn_is_training: True,
                                            self.E_loc_m_avg: E_loc_array})

    def pre_get_amp(self, X0):
        X0 = self.enrich_features(X0)
        return self.sess.run(self.amp, feed_dict={self.x: X0,
                                                  self.bn_is_training: False,})

    def pre_get_log_grads(self, X0):
        X0 = self.enrich_features(X0)
        return self.sess.run(self.log_grads, feed_dict={self.x: X0,
                                                        self.bn_is_training: False,})

    def pre_get_E_grads(self, X0, E_loc_array):
        X0 = self.enrich_features(X0)
        return self.plain_get_E_grads(X0, E_loc_array)

    def getNumPara(self):
        for i in self.para_list:
            print(i.name, i.get_shape().as_list())

        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    def apply_cov_update(self, X0):
        input_dict={}
        input_dict[self.x] = X0
        return self.sess.run(self.cov_update_op, feed_dict=input_dict)

    def apply_inverse_update(self, X0):
        input_dict={}
        input_dict[self.x] = X0
        return self.sess.run(self.inverse_op, feed_dict=input_dict)

    def apply_fisher_inverse(self, F_list, X0):
        input_dict = {i: d for i, d in list(zip(self.newgrads_kfac, F_list))}
        # input_dict[self.x] = X0
        return self.sess.run(self.fisher_inverse, feed_dict=input_dict)
#         return self.sess.run(self.fisher_inverse, feed_dict={i: d for i, d in
#                                                              list(zip(self.newgrads, F_list))})

    def apply_fisher_multiply(self, F_list, X0):
        input_dict = {i: d for i, d in list(zip(self.newgrads_kfac, F_list))}
        # input_dict[self.x] = X0
        return self.sess.run(self.fisher_multiply, feed_dict=input_dict)

    def exp_stabilizer_add(self, increments):
        self.sess.run(self.update_exp_stabilizer, feed_dict={self.dx_exp_stabilizer: increments})

    def build_NN_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = tf.cos(fc1)
            out_re = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_im')
            out = tf.exp(tf.complex(out_re, out_im))

        if self.using_complex:
            return out, tf.complex(out_re, out_im)
        else:
            return tf.real(out), None

    def build_ZNet_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1_amp = tf_.fc_layer(x, self.L, self.L * self.alpha / 2, 'fc1_amp')
            fc1_amp = tf.nn.tanh(fc1_amp)
            fc2_amp = tf_.fc_layer(fc1_amp, self.L * self.alpha / 2, self.L * self.alpha / 2,
                                   'fc2_amp')
            fc2_amp = tf.nn.tanh(fc2_amp)
            out_amp = tf_.fc_layer(fc2_amp, self.L * self.alpha / 2, 1, 'out_amp')
            # out_amp = tf.nn.sigmoid(out_amp)

            fc1_sign = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_sign')
            fc1_sign = tf.cos(fc1_sign)
            fc2_sign = tf_.fc_layer(fc1_sign, self.L * self.alpha, self.L * self.alpha, 'fc2_sign')
            fc2_sign = tf.nn.tanh(fc2_sign)
            out_sign = tf_.fc_layer(fc2_sign, self.L * self.alpha, 1, 'out_sign')
            out_sign = tf.nn.tanh(out_sign)
            out = tf.multiply(out_amp, out_sign)

        if self.using_complex:
            raise NotImplementedError
        else:
            return out, None

    def build_NN_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'fc1')
            fc1 = act(fc1)
            out_re = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_re')
            out_re = tf.clip_by_value(out_re, -60., 60.)
            out_im = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_im')
            log_amp = tf.copmlex(out_re, out_im)
            out = tf.exp(log_amp)

        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_NN_linear_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'fc1')
            fc1 = act(fc1)
            out = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out')

        if self.using_complex:
            raise NotImplementedError
        else:
            return out, None


    def build_NN3_1d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2')
            fc2 = act(fc2)
            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3')
            fc3 = tf.nn.tanh(fc3)
            out_re = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_im')
            log_amp = tf.complex(out_re, out_im)
            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

#     def build_CNN_1d(self, x):
#         with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
#             x = x[:, :, 0:1]
#             inputShape = x.get_shape().as_list()
#             # x_shape = [num_data, Lx, num_spin(channels)]
#             # conv_layer1d(x, filter_size, in_channels, out_channels, name)
#             conv1_re = tf_.conv_layer1d(x, 2, inputShape[-1], self.alpha, 'conv1_re',
#                                         stride_size=2, biases=True)
#             conv1_im = tf_.conv_layer1d(x, 2, inputShape[-1], self.alpha, 'conv1_im',
#                                         stride_size=2, biases=True)
#             conv1 = (tf.complex(conv1_re, conv1_im))
#             # pool4 = tf_.avg_pool2d(conv4, 'pool4', 2)
#             pool4 = tf.reduce_sum(conv1, [1])
#             pool4 = tf_.softplus(pool4)
#
#             # Fully connected layer
# #             fc_dim = self.alpha  # np.prod(pool4.get_shape().as_list()[1:])
# #             pool4 = tf.reshape(pool4, [-1, fc_dim])
# #             out = tf_.fc_layer(pool4, fc_dim, 1, 'out', biases=False, dtype=tf.complex64)
#             # out_re = tf_.fc_layer(pool4, fc_dim, 1, 'out_re', biases=False)
#             # out_im = tf_.fc_layer(pool4, fc_dim, 1, 'out_im', biases=False)
#             out = tf.reduce_sum(pool4, [1], keepdims=True)
#             out_re = tf.real(out)  # + tf_.fc_layer(x[:, :, 0], self.L, 1, 'v_bias_re')
#             out_im = tf.imag(out)
#
#             out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
#
#         return out

    def build_CNN_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2], keepdims=False)
            pool4 = tf.exp(pool4)

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keepdims=False)

            # Fully connected layer
            # fc_dim = self.alpha  # np.prod(pool4.get_shape().as_list()[1:])
            # pool4 = tf.reshape(pool4, [-1, fc_dim])
            # out = tf_.fc_layer(pool4, fc_dim, 1, 'out', biases=False, dtype=tf.complex64)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keepdims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])
            out = tf.reshape(out, [-1, 1])

            # sym_bias = tf_.get_var(tf.truncated_normal([inputShape[1]], 0, 0.1),
            #                        'sym_bias', self.TF_FLOAT)

            # sym_bias = tf.ones([inputShape[1]], self.TF_FLOAT)
            # sym_bias_fft = tf.fft(tf.complex(sym_bias, 0.))
            # x_fft = tf.fft(tf.complex(x[:, :, 0], 0.))
            # sym_phase = tf.real(tf.ifft(x_fft * tf.conj(sym_bias_fft)))
            # theta = tf.scalar_mul(tf.constant(np.pi),
            #                       tf.range(inputShape[1], dtype=self.TF_FLOAT))
            # sym_phase = sym_phase * tf.cos(theta)
            # print(sym_phase.get_shape().as_list())
            # sym_phase = tf.reduce_sum(sym_phase, [1], keepdims=True)
            # print(sym_phase.get_shape().as_list())
            # sym_phase = tf.real(tf.log(tf.complex(sym_phase + 1e-8, 0.)))
            # print(out_im.get_shape().as_list())
            # out_im = tf.add(out_im, sym_phase)
            # print(out_im.get_shape().as_list())

            # out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
            # out = out * tf.exp(tf.complex(0., tf.Variable([1], 1.0, dtype=self.TF_FLOAT)))

        if self.using_complex:
            return out, None
        else:
            return tf.real(out), None

    def build_FCN1_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            x = tf.cast(x, dtype=self.TF_FLOAT)
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300.)
            conv1 = tf_.complex_relu_neg(tf.complex(conv1_re, conv1_im))
            conv2 = conv1

            pool4 = tf.reduce_sum(conv2, [1, 2], keepdims=False)
            pool4 = tf.exp(pool4)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100.)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100.)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keepdims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])

        if self.using_complex:
            return out, None
        else:
            return tf.real(out), None


    def build_FCN2_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            x = tf.cast(x, dtype=self.TF_FLOAT)
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=300.)
            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            conv2 = tf_.circular_conv_1d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=100.)
            conv2 = tf_.softplus2(conv2)

            log_amp = tf.reduce_sum(conv2, [1, 2], keepdims=False)
            pool3 = tf.exp(log_amp)

            ## Conv Bias
            # conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
            #                           [1, 2], keepdims=False)
            # out = tf.reshape(tf.multiply(pool3, tf.exp(conv_bias)), [-1, 1])
            out = tf.reshape(pool3, [-1, 1])
            log_amp = tf.reshape(log_amp, [-1, 1])

        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_FCN3_1d(self, x):
        act = tf_.softplus2
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            x = tf.cast(x, dtype=self.TF_FLOAT)
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1]//4, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1]//4, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300.)
            conv1 = act(tf.complex(conv1_re, conv1_im))
            conv2 = tf_.circular_conv_1d_complex(conv1, inputShape[1]//4, self.alpha, self.alpha,
                                                 'conv2_complex', stride_size=1, biases=True,
                                                 bias_scale=100.)
            conv2 = act(conv2)
            conv3 = tf_.circular_conv_1d_complex(conv2, inputShape[1]//4, self.alpha, self.alpha,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=100.)
            conv3 = act(conv3)

            ## Pooling
            pool4 = tf.reduce_sum(conv3, [1, 2], keepdims=False)
            log_amp = tf.reshape(pool4, [-1, 1])
            out = tf.exp(log_amp)

            ## FC layer
            # conv3 = tf.reduce_sum(conv3, [1], keepdims=False)
            # conv3 = tf.reshape(conv3, [-1, self.alpha*2])
            # out = tf_.fc_layer(conv3, self.alpha*2, 1, 'out_complex',
            #                    biases=True, dtype=tf.complex64)
            # out = tf.reshape(tf.exp(out), [-1])

            ## Conv bias
            # conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
            #                           [1, 2], keepdims=False)
            # conv_bias = tf.exp(conv_bias)
            # out = tf.reshape(tf.multiply(pool4, conv_bias), [-1, 1])

        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_ResNet(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1',
                               dtype=tf.complex64)
            fc1 = act(fc1)
            # fc1 = fc1 + x

            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2',
                               dtype=tf.complex64)
            fc2 = act(fc2)
            fc2 = fc2 + fc1

            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3',
                               dtype=tf.complex64)
            fc3 = act(fc3)
            fc3 = fc3 + fc2

            out = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out',
                               dtype=tf.complex64)
            log_amp = out + tf_.fc_layer(tf.complex(x, 0.), self.L, 1,
                                         'v_bias', dtype=tf.complex64)
            log_amp = tf.reshape(log_amp, [-1, 1])
            out = tf.exp(log_amp)

        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_RBM_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            # inputShape = x.get_shape().as_list()
            x = tf.cast(x[:, :, 0], dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha,
                               'fc1_complex', dtype=tf.complex64)
            # fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            # fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            # fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.softplus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias = tf_.fc_layer(tf.complex(x, 0.), self.L, 1, 'v_bias',
                                  dtype=tf.complex64)
            # v_bias_re = tf_.fc_layer(x, self.L, 1, 'v_bias_re')
            # v_bias_im = tf_.fc_layer(x, self.L, 1, 'v_bias_im')
            # v_bias = tf.complex(v_bias_re, v_bias_im)
            log_prob = tf.reduce_sum(fc2, axis=1, keepdims=True)
            log_prob = tf.add(log_prob, v_bias)
            out = tf.exp(log_prob)

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_RBM_cosh_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            # inputShape = x.get_shape().as_list()
            x = tf.cast(x[:, :, 0], dtype=self.TF_FLOAT) -0.5
            fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)

            v_bias_re = tf_.fc_layer(x, self.L, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, self.L, 1, 'v_bias_im')

            # !!! The implementation below fail for calculting gradient !!!
            # !!! forward prediction is correct !!!
            # fc2 = tf.cosh(fc1)
            # v_bias = tf.exp(tf.complex(v_bias_re, v_bias_im))
            # out = tf.multiply(v_bias, tf.reduce_prod(fc2, axis=1, keepdims=True))
            # out = tf.real(out)
            fc2 = tf.log(tf.cosh(fc1))
            log_prob = tf.reduce_sum(fc2, axis=1, keepdims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.exp(log_prob)

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_sRBM_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2], keepdims=False)
            pool4 = tf.exp(pool4)

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keepdims=False)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keepdims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])

        if self.using_complex:
            return out, None
        else:
            return tf.real(out), None

    def build_NN_complex(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha,
                                       'fc1_complex', dtype=tf.complex64, biases=True)
            fc1_complex = act(fc1_complex)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, 1, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)
            fc2_complex = tf.reshape(fc2_complex, [-1, 1])
            out = tf.exp(fc2_complex)
            # out = (fc2_complex)
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return out, fc2_complex
        else:
            return tf.real(out), None

    def build_NN3_complex(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1_complex',
                                       dtype=tf.complex64)
            fc1_complex = act(fc1_complex)
            fc1_complex = fc1_complex + tf.complex(x, 0.)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, self.L * self.alpha, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)
            fc2_complex = act(fc2_complex)
            fc2_complex = fc2_complex + fc1_complex

            fc3_complex = tf_.fc_layer(fc2_complex, self.L * self.alpha, self.L * self.alpha, 'fc3_complex',
                                       dtype=tf.complex64, biases=True)
            fc3_complex = act(fc3_complex)
            fc3_complex = fc3_complex  + fc2_complex

            fc4_complex = tf.reduce_sum(fc3_complex, axis=1, keepdims=True)
            fc4_complex = tf.reshape(fc4_complex, [-1, 1])
            # out = tf.multiply(tf.exp(tf.real(fc4_complex) / self.L), tf.cos(tf.imag(fc4_complex)))
            # fc4_complex = tf_.fc_layer(fc3_complex, self.L * self.alpha, 1, 'fc4_complex',
            #                            dtype=tf.complex64, biases=True)
            out = tf.exp(fc4_complex)
            # out = (fc4_complex)
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return out, None
        else:
            return tf.real(out), None

    def build_RBM_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            inputShape = x.get_shape().as_list()
            b_size, Lx, Ly, _ = inputShape
            LxLy = Lx * Ly
            x = tf.reshape(x[:, :, :, 0], [-1, LxLy])
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1_re = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_re', dtype=self.TF_FLOAT)
            fc1_im = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_im', dtype=self.TF_FLOAT)
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.softplus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re', dtype=self.TF_FLOAT)
            v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im', dtype=self.TF_FLOAT)
            log_prob = tf.reduce_sum(fc2, axis=1, keepdims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.exp(log_prob)

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_RBM_cosh_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            inputShape = x.get_shape().as_list()
            x = tf.cast(x, dtype=self.TF_FLOAT)
            b_size, Lx, Ly, _ = inputShape
            LxLy = Lx * Ly
            x = tf.reshape(x[:, :, :, 0], [-1, LxLy])
            fc1_re = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_im') * 100
            fc1 = tf.complex(fc1_re, fc1_im)
            # fc2 = tf_.complex_relu(fc1)

            # fc2 = tf.cosh(fc1)
            # v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re')
            # v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im') * 100
            # v_bias = tf.exp(tf.complex(v_bias_re, v_bias_im))
            # out = tf.multiply(v_bias, tf.reduce_prod(fc2, axis=1, keepdims=True))
            # out = tf.real(out)

            fc2 = tf.log(tf.cosh(fc1))
            v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keepdims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            log_prob = tf.reshape(log_prob, [-1,1])
            out = tf.exp(log_prob)

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_sRBM_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered, dtype=self.TF_FLOAT)
                                            # stride_size=2, biases=True, bias_scale=3000.*2/64/self.alpha, FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered, dtype=self.TF_FLOAT)
                                            # stride_size=2, biases=True, bias_scale=3140.*2/64/self.alpha, FFT=False)

            conv1 = act(tf.complex(conv1_re, conv1_im))
            # conv1 = tf_.complex_relu(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2, 3], keepdims=False)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = tf.real(pool4)
            pool4_imag = tf.imag(pool4)
            # pool4 = tf.exp(tf.complex(pool4_real, pool4_imag))
            # pool4 = tf.exp(pool4)
            # out = tf.real((out))
            # return tf.reshape(tf.real(pool4), [-1,1])

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keepdims=False)

            conv_bias_re = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False,
                                                layer_collection=self.layer_collection,
                                                registered=self.registered, dtype=self.TF_FLOAT)
            conv_bias_im = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False,
                                                layer_collection=self.layer_collection,
                                                registered=self.registered, dtype=self.TF_FLOAT)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2, 3], keepdims=False)
            final_real = tf.clip_by_value(pool4_real + tf.real(conv_bias), -60., 60.)
            final_imag = pool4_imag + tf.imag(conv_bias)
            final_real = final_real - self.exp_stabilizer
            out = tf.reshape(tf.exp(tf.complex(final_real, final_imag)), [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, None
        else:
            return tf.real(out), None

    def build_pre_sRBM_2d(self, x):
        '''
        the input x is with features:
        before one-hot encoding
            0-1: spin configuration
            2: dense positive cluster size, cutoff for > 5
            3: dense negative cluster size, cutoff for > 5
        after one-hot encoding
            0-1: spin configuration
            2-6: positive cluster size
            7-11: negative cluster size
        '''
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            inputShape = x.get_shape().as_list()
            # minus one, so we only encode from 1
            dense_pos_cluster_size = x[:, :, :, 2] - 1
            dense_neg_cluster_size = x[:, :, :, 3] - 1
            pos_cluster_size_layer = tf.one_hot(dense_pos_cluster_size, depth=5, axis=-1)
            neg_cluster_size_layer = tf.one_hot(dense_neg_cluster_size, depth=5, axis=-1)
            x = tf.concat([tf.cast(x[:,:,:,:2], dtype=self.TF_FLOAT), pos_cluster_size_layer, neg_cluster_size_layer],
                          axis=-1)
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], 12, self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False)
                                            # stride_size=2, biases=True, bias_scale=3000.*2/64/self.alpha, FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], 12, self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False)
                                            # stride_size=2, biases=True, bias_scale=3140.*2/64/self.alpha, FFT=False)

            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            # conv1 = tf_.complex_relu(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2, 3], keepdims=False)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = tf.real(pool4)
            pool4_imag = tf.imag(pool4)
            # pool4 = tf.exp(tf.complex(pool4_real, pool4_imag))
            # pool4 = tf.exp(pool4)
            # out = tf.real((out))
            # return tf.reshape(tf.real(pool4), [-1,1])

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keepdims=False)

            conv_bias_re = tf_.circular_conv_2d(x[:, :, :, 0:2], 2, 2, 1, 'conv_bias_re',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias_im = tf_.circular_conv_2d(x[:, :, :, 0:2], 2, 2, 1, 'conv_bias_im',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2, 3], keepdims=False)
            final_real = pool4_real + tf.real(conv_bias)
            # final_real = tf.clip_by_value(final_real, -60., 60.)
            final_real = final_real - self.exp_stabilizer
            final_imag = pool4_imag + tf.imag(conv_bias)
            log_prob = tf.reshape(tf.complex(final_real, final_imag), [-1, 1])
            out = tf.exp(log_prob)

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_FCN2v0_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv2 = act(conv2)

            pool3 = tf.reduce_sum(conv2[:, :, :, :self.alpha], [1, 2, 3], keepdims=False) -\
                    tf.reduce_sum(conv2[:, :, :, self.alpha:], [1, 2, 3], keepdims=False)

            pool3_real = tf.real(pool3)
            # pool3_real = tf.clip_by_value(tf.real(pool3), -70., 70.)
            # pool3_real = tf.real(pool3) - self.exp_stabilizer
            pool3_imag = tf.imag(pool3)
            log_prob = tf.complex(pool3_real,pool3_imag)
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_real_FCN2_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1 = act(conv1_re)
            conv2_re = tf_.circular_conv_2d(conv1, inputShape[1], self.alpha, self.alpha, 'conv2_re',
                                            stride_size=2, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv2_im = tf_.circular_conv_2d(conv1, inputShape[1], self.alpha, self.alpha, 'conv2_im',
                                            stride_size=2, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv2 = act(tf.complex(conv2_re, conv2_im))

            pool3 = tf.reduce_sum(conv2, [1, 2, 3], keepdims=False)
            pool3_real = tf.real(pool3)
            # pool3_real = tf.clip_by_value(tf.real(pool3), -60., 60.)
            pool3_imag = tf.imag(pool3)
            log_prob = tf.complex(pool3_real, pool3_imag)
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_FCN2_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1], self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv2 = act(conv2)

            pool3 = tf.reduce_sum(conv2, [1, 2, 3], keepdims=False)
            pool3_real = tf.real(pool3)
            # pool3_real = tf.clip_by_value(tf.real(pool3), -60., 60.)
            pool3_imag = tf.imag(pool3)
            log_prob = tf.complex(pool3_real,pool3_imag)
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_FCN3v0_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv2 = act(conv2)

            conv3 = tf_.circular_conv_2d_complex(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv3 = act(conv3)

            # pool4 = tf.reduce_sum(conv3[:, :, :, :self.alpha], [1, 2, 3], keepdims=False) -\
            #         tf.reduce_sum(conv3[:, :, :, self.alpha:], [1, 2, 3], keepdims=False)
            pool4 = tf.reduce_sum(conv3, [1, 2, 3], keepdims=False)
            pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_imag = tf.imag(pool4)
            log_prob = tf.complex(pool4_real, pool4_imag)
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.exp(log_prob)

            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_FCN3v2_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False,
                                            layer_collection=self.layer_collection,
                                            registered=self.registered)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv2 = act(conv2)

            conv3 = tf_.circular_conv_2d_complex(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=1., layer_collection=self.layer_collection,
                                                 registered=self.registered)
            conv3 = act(conv3)

            pool4 = tf.reduce_sum(conv3, [1, 2, 3], keepdims=False)
            pool4_real = tf.real(pool4)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = pool4_real - self.exp_stabilizer
            pool4_imag = tf.imag(pool4)
            log_prob = tf.complex(pool4_real, pool4_imag)
            out = tf.exp(log_prob)

            out = tf.reshape(out, [-1, 1])
            log_prob = tf.reshape(log_prob, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_NN3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha//2, 'fc1',
                               layer_collection=self.layer_collection, registered=self.registered)
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc2',
                               layer_collection=self.layer_collection, registered=self.registered)
            fc2 = act(fc2)
            fc3 = tf_.fc_layer(fc2, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc3',
                               layer_collection=self.layer_collection, registered=self.registered)
            fc3 = act(fc3)
            out_re = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_re',
                                  layer_collection=self.layer_collection, registered=self.registered)
            # out_re = tf.clip_by_value(out_re, -60., 60.)
            out_im = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_im',
                                  layer_collection=self.layer_collection, registered=self.registered)
            log_amp = tf.complex(out_re, out_im)
            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_MADE_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x_reshaped = tf.reshape(x, [-1, self.LxLy, 2])
            x = x_reshaped[:, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1 = tf_.masked_fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'masked_fc1',
                                      self.ordering, 'A', layer_collection=self.layer_collection,
                                      registered=self.registered, dtype=self.TF_FLOAT)
            fc1 = act(fc1)
            fc2 = tf_.masked_fc_layer(fc1, self.LxLy * self.alpha, self.LxLy * self.alpha,
                                      'masked_fc2', self.ordering, 'B',
                                      layer_collection=self.layer_collection,
                                      registered=self.registered, dtype=self.TF_FLOAT)
            fc2 = act(fc2)
            fc3 = tf_.masked_fc_layer(fc2, self.LxLy * self.alpha, self.LxLy * 4,
                                      'masked_fc3', self.ordering, 'B',
                                      layer_collection=self.layer_collection,
                                      registered=self.registered, dtype=self.TF_FLOAT)

            fc3 = tf.reshape(fc3, [-1, self.LxLy, 4])
            out0_re = fc3[:,:,0]
            out1_re = fc3[:,:,1]
            out0_im = fc3[:,:,2]
            out1_im = fc3[:,:,3]
            ## stable normalize ##
            max_re = tf.math.maximum(out0_re, out1_re)
            out0_re = out0_re - max_re
            out1_re = out1_re - max_re
            log_l2_norm = tf.log(tf.exp(2*out0_re) + tf.exp(2*out1_re)) / 2.
            out0_re = out0_re - log_l2_norm
            out1_re = out1_re - log_l2_norm

            log_cond_amp_0 = tf.complex(out0_re, out0_im)
            log_cond_amp_1 = tf.complex(out1_re, out1_im)
            log_cond_amp = tf.stack([log_cond_amp_0, log_cond_amp_1], axis=-1)
            ## now a complex tensor of shape [batch_size, LxLy, 2]

            ############################################################
            ### Constructed a path without involving complex number ####
            ############################################################
            re_cond_amp = tf.stack([2 * out0_re, 2 * out0_im], axis=-1)
            log_prob = tf.reduce_sum(tf.multiply(
                re_cond_amp, tf.cast(x_reshaped, self.TF_FLOAT)),
                                     axis=[1, 2])
            prob = tf.exp(log_prob)

            log_amp = tf.reduce_sum(tf.multiply(
                log_cond_amp, tf.cast(x_reshaped, self.TF_COMPLEX)),
                                    axis=[1, 2])

            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        self.registered = True
        if self.using_complex:
            return out, log_amp, log_cond_amp, prob
        else:
            return tf.real(out), None, log_cond_amp, prob

    def build_pixelCNNv2_2d(self, x, activation, num_blocks,
                            weight_normalization=True):
        assert (self.using_complex)
        pixel_block_sharir_v2 = tf_.pixel_block_sharir_v2
        pixel_resiual_block = tf_.pixel_resiual_block
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x_reshaped = tf.reshape(x, [-1, self.LxLy, self.channels])

            pixel_input = tf.cast(x, dtype=self.TF_FLOAT)
            px = pixel_block_sharir_v2(x=pixel_input,
                                       in_channel=self.channels,
                                       out_channel=8 * self.alpha,
                                       block_type='start',
                                       name='pixel_0',
                                       dtype=self.TF_FLOAT,
                                       activation=act,
                                       layer_collection=self.layer_collection,
                                       registered=self.registered,
                                       weight_normalization=weight_normalization,
                                      )
            for idx in range(num_blocks//2):
                px = pixel_resiual_block(px, 'res_block'+str(idx), self.TF_FLOAT, filter_size=3,
                                         activation=act, num_of_layers=2,
                                         layer_collection=self.layer_collection,
                                         registered=self.registered,
                                         weight_normalization=weight_normalization)

            px = pixel_block_sharir_v2(x=px,
                                       in_channel=8 * self.alpha,
                                       out_channel=8 * self.alpha,
                                       block_type='mid',
                                       name='pixel_end',
                                       dtype=self.TF_FLOAT,
                                       activation=act,
                                       layer_collection=self.layer_collection,
                                       registered=self.registered,
                                       weight_normalization=weight_normalization,
                                      )
            hor_x = px[:,:,:,4 * self.alpha:]
            px = tf_.conv_layer2d(hor_x,
                                  1,
                                  4 * self.alpha,
                                  self.channels * 2,
                                  'end_conv',
                                  dtype=self.TF_FLOAT,
                                  padding='VALID',
                                  weight_normalization=weight_normalization,
                                  layer_collection=self.layer_collection,
                                  registered=self.registered)


            conv_out = tf.reshape(px, [-1, self.LxLy, self.channels * 2])
            log_cond_amp_re_list = []
            log_cond_amp_im_list = []
            for i in range(self.channels):
                log_cond_amp_re_list.append(conv_out[:, :, i])
                log_cond_amp_im_list.append(conv_out[:, :, self.channels + i])

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            max_re = tf.math.reduce_max(log_cond_amp_re_stacked, axis=-1)
            for i in range(self.channels):
                log_cond_amp_re_list[i] = log_cond_amp_re_list[i] - max_re

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            log_l2_norm = tf.log(
                tf.math.reduce_sum(tf.exp(2 * log_cond_amp_re_stacked),
                                   axis=[-1])) / 2.
            for i in range(self.channels):
                log_cond_amp_re_list[i] = log_cond_amp_re_list[i] - log_l2_norm

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            log_cond_amp_im_stacked = tf.stack(log_cond_amp_im_list, axis=-1)
            log_cond_amp = tf.complex(log_cond_amp_re_stacked,
                                      log_cond_amp_im_stacked)
            # now a complex tensor of shape [batch_size, LxLy, num_channels]

            ############################################################
            ### Constructed a path without involving complex number ####
            ############################################################
            # re_cond_prob = tf.stack([2*out0_re, 2*out1_re], axis=-1)
            re_cond_prob = log_cond_amp_re_stacked * 2.

            log_prob = tf.reduce_sum(tf.multiply(
                re_cond_prob, tf.cast(x_reshaped, self.TF_FLOAT)),
                                     axis=[1, 2])
            prob = tf.exp(log_prob)

            log_amp = tf.reduce_sum(tf.multiply(
                log_cond_amp, tf.cast(x_reshaped, self.TF_COMPLEX)),
                                    axis=[1, 2])

            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        if not self.using_symm:
            self.registered = True
            return out, log_amp, log_cond_amp, prob
        else:
            raise




    def build_pixelCNN_2d(self, x, activation, num_blocks, mode,
                          residual_connection=False,
                          BN=False,
                          split_block=False,
                          weight_normalization=False,
                         ):
        assert (self.using_complex)
        if mode == '1':
            pixel_block = tf_.pixel_block
        elif mode == '2':
            pixel_block = tf_.pixel_block_sharir
        elif mode == '3':
            pixel_block = tf_.pixel_block_resnext

        act = tf_.select_activation(activation)
        #######################################
        ## single model, i.e. w.o. symmetry
        ## using for sampling perpose only
        #######################################
        self.registered = self.registered or self.using_symm  # Not to register the sampling head
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x_reshaped = tf.reshape(x, [-1, self.LxLy, self.channels])

            pixel_input = tf.cast(x, dtype=self.TF_FLOAT)
            px = pixel_block(
                pixel_input,
                self.channels,
                8 * self.alpha,
                'start',
                'pixel_0',
                self.TF_FLOAT,
                activation=act,
                layer_collection=self.layer_collection,
                registered=self.registered,
                residual_connection=residual_connection,
                weight_normalization=weight_normalization,
                BN=BN, bn_phase=self.bn_is_training,
                split_block=split_block,
            )
            for i in range(1, num_blocks):
                px = pixel_block(
                    px,
                    8 * self.alpha,
                    8 * self.alpha,
                    'mid',
                    'pixel_' + str(i),
                    self.TF_FLOAT,
                    activation=act,
                    layer_collection=self.layer_collection,
                    registered=self.registered,
                    residual_connection=residual_connection,
                    weight_normalization=weight_normalization,
                    BN=BN, bn_phase=self.bn_is_training,
                    split_block=split_block,
                )

            px = pixel_block(
                px,
                8 * self.alpha,
                self.channels * 2,
                'end',
                'pixel_end',
                self.TF_FLOAT,
                activation=act,
                layer_collection=self.layer_collection,
                registered=self.registered,
                residual_connection=residual_connection,
                weight_normalization=weight_normalization,
                BN=BN, bn_phase=self.bn_is_training,
                split_block=split_block,
            )

            fc3 = tf.reshape(px, [-1, self.LxLy, self.channels * 2])

            conserved_Sz = True
            if conserved_Sz:
                assert self.channels == 2
                np_mask = mask.gen_fc_mask(self.ordering,
                                           mask_type='A',
                                           dtype=self.NP_FLOAT,
                                           in_hidden=1,
                                           out_hidden=1)
                tf_mask = tf.constant(np_mask, dtype=self.TF_FLOAT)
                # x_reshaped = tf.reshape(x, [-1, self.LxLy, self.channels])
                num_c0 = tf.matmul(tf.cast(x_reshaped[:, :, 0], self.TF_FLOAT),
                                   tf_mask)
                # p_c0 = 1 if LxLy / 2 - 0.001 - num_c0 > 0
                p_c0 = tf.math.sign(-num_c0 - 0.001 + self.LxLy // 2 )
                p_c0 = tf.nn.relu(p_c0) + 1e-36
                num_c1 = tf.matmul(tf.cast(x_reshaped[:, :, 1], self.TF_FLOAT),
                                   tf_mask)
                p_c1 = tf.math.sign(-num_c1 - 0.001 + self.LxLy // 2 )
                p_c1 = tf.nn.relu(p_c1) + 1e-36
                ## We add small value epsilon ~= 1e-36 to the probability
                ## to avoid the numerical undefined value p * log p, when
                ## p --> 0
                ## THE IS IMPORTANT FOR THE CODE TO WORK !!!
                ##
                log_p_c0 = tf.log(p_c0)
                log_p_c1 = tf.log(p_c1)
                log_p_symm_constrain = tf.stack([log_p_c0, log_p_c1], axis=-1)
                ## Although this is the constrain on probability
                ## since, we only consider the case of having 1 or 0 in probability
                ## this corresponds to probability amplitude having 0 and -inf
                ## in the real part.
                ## Therefore, we directly add this value to the real part, before normalizing it.
                ##
                fc3 = tf.concat([log_p_symm_constrain + fc3[:,:,:self.channels],
                                 fc3[:,:,self.channels:]], axis=-1)



            # out0_re = fc3[:,:,0]
            # out1_re = fc3[:,:,1]
            # out0_im = fc3[:,:,2]
            # out1_im = fc3[:,:,3]
            # ## stable normalize ##
            # max_re = tf.math.maximum(out0_re, out1_re)
            # out0_re = out0_re - max_re
            # out1_re = out1_re - max_re
            # log_l2_norm = tf.log(tf.exp(2*out0_re) + tf.exp(2*out1_re)) / 2.
            # out0_re = out0_re - log_l2_norm
            # out1_re = out1_re - log_l2_norm

            # log_cond_amp_0 = tf.complex(out0_re, out0_im)
            # log_cond_amp_1 = tf.complex(out1_re, out1_im)
            # log_cond_amp = tf.stack([log_cond_amp_0, log_cond_amp_1], axis=-1)
            # ## now a complex tensor of shape [batch_size, LxLy, 2]

            log_cond_amp_re_list = []
            log_cond_amp_im_list = []
            for i in range(self.channels):
                log_cond_amp_re_list.append(fc3[:, :, i])
                log_cond_amp_im_list.append(fc3[:, :, self.channels + i])

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            max_re = tf.math.reduce_max(log_cond_amp_re_stacked, axis=-1)
            for i in range(self.channels):
                log_cond_amp_re_list[i] = log_cond_amp_re_list[i] - max_re

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            log_l2_norm = tf.log(
                tf.math.reduce_sum(tf.exp(2 * log_cond_amp_re_stacked),
                                   axis=[-1])) / 2.
            for i in range(self.channels):
                log_cond_amp_re_list[i] = log_cond_amp_re_list[i] - log_l2_norm

            log_cond_amp_re_stacked = tf.stack(log_cond_amp_re_list, axis=-1)
            log_cond_amp_im_stacked = tf.stack(log_cond_amp_im_list, axis=-1)
            log_cond_amp = tf.complex(log_cond_amp_re_stacked,
                                      log_cond_amp_im_stacked)
            # now a complex tensor of shape [batch_size, LxLy, num_channels]

            ############################################################
            ### Constructed a path without involving complex number ####
            ############################################################
            # re_cond_prob = tf.stack([2*out0_re, 2*out1_re], axis=-1)
            re_cond_prob = log_cond_amp_re_stacked * 2.

            log_prob = tf.reduce_sum(tf.multiply(
                re_cond_prob, tf.cast(x_reshaped, self.TF_FLOAT)),
                                     axis=[1, 2])
            prob = tf.exp(log_prob)

            log_amp = tf.reduce_sum(tf.multiply(
                log_cond_amp, tf.cast(x_reshaped, self.TF_COMPLEX)),
                                    axis=[1, 2])

            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        if not self.using_symm:
            self.registered = True
            return out, log_amp, log_cond_amp, prob
        else:
            #######################################
            ## mixture model, i.e. w. symmetry  ###
            #######################################
            self.registered = False
            with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
                num_symm = 4
                symm_x = tf.concat([
                    tf.image.rot90(x, k=0),
                    tf.image.rot90(x, k=1),
                    tf.image.rot90(x, k=2),
                    tf.image.rot90(x, k=3),
#                     tf.image.rot90(1 - x, k=0),
#                     tf.image.rot90(1 - x, k=1),
#                     tf.image.rot90(1 - x, k=2),
#                     tf.image.rot90(1 - x, k=3)
                ],
                                   axis=0)

                symm_x_reshaped = tf.reshape(symm_x,
                                             [-1, self.LxLy, self.channels])

                symm_pixel_input = tf.cast(symm_x, dtype=self.TF_FLOAT)
                px = pixel_block(
                    symm_pixel_input,
                    self.channels,
                    8 * self.alpha,
                    'start',
                    'pixel_0',
                    self.TF_FLOAT,
                    activation=act,
                    layer_collection=self.layer_collection,
                    registered=self.registered,
                    residual_connection=residual_connection,
                    weight_normalization=weight_normalization,
                    BN=BN, bn_phase=self.bn_is_training,
                    split_block=split_block,
                )
                for i in range(1, num_blocks):
                    px = pixel_block(
                        px,
                        8 * self.alpha,
                        8 * self.alpha,
                        'mid',
                        'pixel_' + str(i),
                        self.TF_FLOAT,
                        activation=act,
                        layer_collection=self.layer_collection,
                        registered=self.registered,
                        residual_connection=residual_connection,
                        weight_normalization=weight_normalization,
                        BN=BN, bn_phase=self.bn_is_training,
                        split_block=split_block,
                    )

                px = pixel_block(
                    px,
                    8 * self.alpha,
                    self.channels * 2,
                    'end',
                    'pixel_end',
                    self.TF_FLOAT,
                    activation=act,
                    layer_collection=self.layer_collection,
                    registered=self.registered,
                    residual_connection=residual_connection,
                    weight_normalization=weight_normalization,
                    BN=BN, bn_phase=self.bn_is_training,
                    split_block=split_block,
                )

                symm_fc3 = tf.reshape(px, [-1, self.LxLy, self.channels * 2])

                # symm_out0_re = symm_fc3[:,:,0]
                # symm_out1_re = symm_fc3[:,:,1]
                # symm_out0_im = symm_fc3[:,:,2]
                # symm_out1_im = symm_fc3[:,:,3]
                # ## stable normalize ##
                # symm_max_re = tf.math.maximum(symm_out0_re, symm_out1_re)
                # symm_out0_re = symm_out0_re - symm_max_re
                # symm_out1_re = symm_out1_re - symm_max_re
                # symm_log_l2_norm = tf.log(tf.exp(2*symm_out0_re) + tf.exp(2*symm_out1_re)) / 2.
                # symm_out0_re = symm_out0_re - symm_log_l2_norm
                # symm_out1_re = symm_out1_re - symm_log_l2_norm

                # symm_log_cond_amp_0 = tf.complex(symm_out0_re, symm_out0_im)
                # symm_log_cond_amp_1 = tf.complex(symm_out1_re, symm_out1_im)
                # symm_log_cond_amp = tf.stack([symm_log_cond_amp_0,
                #                               symm_log_cond_amp_1], axis=-1)
                # ## now a complex tensor of shape [batch_size, LxLy, 2]


                symm_log_cond_amp_re_list = []
                symm_log_cond_amp_im_list = []
                for i in range(self.channels):
                    symm_log_cond_amp_re_list.append(symm_fc3[:,:,i])
                    symm_log_cond_amp_im_list.append(symm_fc3[:,:,self.channels+i])

                symm_log_cond_amp_re_stacked = tf.stack(symm_log_cond_amp_re_list, axis=-1)
                symm_max_re = tf.math.reduce_max(symm_log_cond_amp_re_stacked, axis=-1)
                for i in range(self.channels):
                    symm_log_cond_amp_re_list[i] = symm_log_cond_amp_re_list[i] - symm_max_re

                symm_log_cond_amp_re_stacked = tf.stack(symm_log_cond_amp_re_list, axis=-1)
                symm_log_l2_norm = tf.log(tf.math.reduce_sum(tf.exp(2*symm_log_cond_amp_re_stacked),
                                                             axis=[-1])) / 2.
                for i in range(self.channels):
                    symm_log_cond_amp_re_list[i] = symm_log_cond_amp_re_list[i] - symm_log_l2_norm

                symm_log_cond_amp_re_stacked = tf.stack(symm_log_cond_amp_re_list, axis=-1)
                symm_log_cond_amp_im_stacked = tf.stack(symm_log_cond_amp_im_list, axis=-1)
                symm_log_cond_amp = tf.complex(log_cond_amp_re_stacked, log_cond_amp_im_stacked)
                # now a complex tensor of shape [batch_size, LxLy, num_channels]


                ############################################################
                ### Constructed a path without involving complex number ####
                ############################################################
                # symm_re_cond_prob = tf.stack([2*symm_out0_re, 2*symm_out1_re], axis=-1)
                symm_re_cond_prob = symm_log_cond_amp_re_stacked * 2.

                symm_log_prob = tf.reduce_sum(tf.multiply(symm_re_cond_prob,
                                                          tf.cast(symm_x_reshaped, self.TF_FLOAT)), axis=[1,2])
                symm_log_prob = tf.transpose(tf.reshape(symm_log_prob, [num_symm, -1]), [1,0])
                symm_prob = tf.exp(symm_log_prob)
                symm_prob = tf.reduce_sum(symm_prob, axis=[1]) / float(num_symm)

                # symm_im_cond_amp = tf.stack([tf.complex(tf.zeros_like(symm_out0_im, dtype=self.TF_FLOAT),
                #                                         symm_out0_im),
                #                              tf.complex(tf.zeros_like(symm_out1_im, dtype=self.TF_FLOAT),
                #                                         symm_out1_im)], axis=-1)
                # symm_im_log_amp = tf.reduce_sum(tf.multiply(symm_im_cond_amp,
                #                                             tf.cast(symm_x_reshaped, self.TF_COMPLEX)), axis=[1,2])
                symm_im_log_amp = tf.reduce_sum(tf.multiply(symm_log_cond_amp_im_stacked,
                                                            tf.cast(symm_x_reshaped, self.TF_FLOAT)), axis=[1,2])
                symm_im_log_amp = tf.complex(tf.zeros_like(symm_im_log_amp), symm_im_log_amp)


                symm_im_log_amp = tf.transpose(tf.reshape(symm_im_log_amp, [num_symm, -1]), [1,0])
                symm_im_amp = tf.exp(symm_im_log_amp)
                symm_im_amp = tf.reduce_sum(symm_im_amp, axis=[1])

                final_symm_log_amp_re = tf.log(symm_prob) / 2.
                final_symm_log_amp_im = tf.imag(tf.log(symm_im_amp))
                final_symm_log_amp = tf.complex(final_symm_log_amp_re, final_symm_log_amp_im)
                final_symm_log_amp = tf.reshape(final_symm_log_amp, [-1, 1])
                final_symm_out = tf.exp(final_symm_log_amp)

            self.registered=True

            if self.using_complex:
                return (out, log_amp, log_cond_amp, prob,
                        final_symm_out, final_symm_log_amp, None, symm_prob)
            else:
                raise NotImplementedError
                # return tf.real(out), None, log_cond_amp, prob


#     def build_NADE_2d(self, x, activation):
#     def build_pixelCNN_2d(self, x, activation):
#         act = tf_.select_activation(activation)
#         with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
#             x_reshaped = tf.reshape(x, [-1, self.LxLy, 2])
#             x_input = tf.cast(x, dtype=self.TF_FLOAT)
#             cv1 = tf_.masked_conv_layer2d(x_input, 5, 2, self.alpha*4, 'A',
#                                           'masked_conv1', dtype=self.TF_FLOAT,
#                                           layer_collection=self.layer_collection,
#                                           registered=self.registered)
#             cv1 = act(cv1)
#             cv2 = tf_.masked_conv_layer2d(cv1, 3, self.alpha*4, self.alpha*2,
#                                           'B', 'masked_conv2', dtype=self.TF_FLOAT,
#                                           layer_collection=self.layer_collection,
#                                           registered=self.registered)
#             cv2 = act(cv2)
#             cv3 = tf_.masked_conv_layer2d(cv2, 3, self.alpha*2, 4,
#                                           'B', 'masked_conv3', dtype=self.TF_FLOAT,
#                                           layer_collection=self.layer_collection,
#                                           registered=self.registered)
#             fc3 = tf.reshape(cv3, [-1, self.LxLy, 4])
# 
#             x_input2 = x_reshaped[:, :, 0]
#             x_input2 = tf.cast(x_input2, dtype=self.TF_FLOAT)
#             fc1 = tf_.masked_fc_layer(x_input2, self.LxLy, 2*self.LxLy, 'masked_fc1',
#                                       self.ordering, 'A', layer_collection=self.layer_collection,
#                                       registered=self.registered, dtype=self.TF_FLOAT)
#             fc1 = act(fc1)
#             fc1 = tf.reshape(fc1, [-1, self.LxLy, 2])
#             # fc3 = fc3 - fc1
# 
#             # cv3 = act(cv3)
#             # fc3 = tf.reshape(cv3, [-1, self.LxLy, self.alpha])
#             # fc3 = tf_.masked_fc_layer(fc3, self.LxLy * self.alpha, self.LxLy * 4,
#             #                           'masked_fc3', self.ordering, 'B', dtype=self.TF_FLOAT,
#             #                           layer_collection=self.layer_collection,
#             #                           registered=self.registered)
#             # fc3 = tf.reshape(fc3, [-1, self.LxLy, 4])
# 
#             out0_re = fc3[:,:,0] + fc1[:,:,0]
#             out1_re = fc3[:,:,1] + fc1[:,:,1]
#             out0_im = fc3[:,:,2]
#             out1_im = fc3[:,:,3]
#             ## stable normalize ##
#             max_re = tf.math.maximum(out0_re, out1_re)
#             out0_re = out0_re - max_re
#             out1_re = out1_re - max_re
#             log_l2_norm = tf.log(tf.exp(2*out0_re) + tf.exp(2*out1_re)) / 2.
#             out0_re = out0_re - log_l2_norm
#             out1_re = out1_re - log_l2_norm
# 
#             log_cond_amp_0 = tf.complex(out0_re, out0_im)
#             log_cond_amp_1 = tf.complex(out1_re, out1_im)
#             log_cond_amp = tf.stack([log_cond_amp_0, log_cond_amp_1], axis=-1)
#             ## now a complex tensor of shape [batch_size, LxLy, 2]
# 
#             ############################################################
#             ### Constructed a path without involving complex number ####
#             ############################################################
#             re_cond_amp = tf.stack([2*out0_re, 2*out0_im], axis=-1)
#             log_prob = tf.reduce_sum(tf.multiply(re_cond_amp, tf.cast(x_reshaped, self.TF_FLOAT)), axis=[1,2])
#             prob = tf.exp(log_prob)
# 
#             log_amp = tf.reduce_sum(tf.multiply(log_cond_amp, tf.cast(x_reshaped, self.TF_COMPLEX)), axis=[1,2])
# 
#             out = tf.exp(log_amp)
#             out = tf.reshape(out, [-1, 1])
# 
#         self.registered=True
#         if self.using_complex:
#             return out, log_amp, log_cond_amp, prob
#         else:
#             return tf.real(out), None, log_cond_amp, prob



    def build_ResNN3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            x = tf.reshape(x, [-1, self.LxLy])
            p_fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha//2, 'fc1',
                                 layer_collection=self.layer_collection, registered=self.registered)
            fc1 = act(p_fc1)
            fc1 = fc1 + x
            p_fc2 = tf_.fc_layer(fc1, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc2',
                                 layer_collection=self.layer_collection, registered=self.registered)
            fc2 = act(p_fc2)
            fc2 = fc2 + fc1
            p_fc3 = tf_.fc_layer(fc2, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc3',
                                 layer_collection=self.layer_collection, registered=self.registered)
            fc3 = act(p_fc3)
            fc3 = fc3 + fc2
            out_re = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_re',
                                  layer_collection=self.layer_collection, registered=self.registered)
            # out_re = tf.clip_by_value(out_re, -60., 60.)
            out_im = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_im',
                                  layer_collection=self.layer_collection, registered=self.registered)
            log_amp = tf.complex(out_re, out_im)
            out = tf.exp(log_amp)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_amp
        else:
            return tf.real(out), None

    def build_real_CNN_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1 = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1',
                                         stride_size=2, biases=True, bias_scale=1., FFT=False,
                                         layer_collection=self.layer_collection,
                                         registered=self.registered)
            conv1 = act(conv1)
            pool1 = tf.reduce_mean(conv1, [1, 2], keepdims=False)
            # pool1 = tf.Print(pool1,[pool1[:3,:], 'pool1'])

            out_re = tf_.fc_layer(pool1, self.alpha, 1, 'out_re', biases=False,
                                  layer_collection=self.layer_collection, registered=self.registered)
            out_re = tf.clip_by_value(out_re, -60., 60.)
            # out_re = tf.Print(out_re, [out_re[:3,:], 'out_re'])

            out_im = tf_.fc_layer(pool1, self.alpha, 1, 'out_im', biases=False,
                                  layer_collection=self.layer_collection, registered=self.registered)
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            out = tf.multiply(tf.exp(out_re), tf.sin(out_im))
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            raise NotImplementedError
            return out, tf.complex(out_re, math.pi/2.-out_im)
        else:
            return out, None

    def build_real_CNN3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1 = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1',
                                         stride_size=1, biases=True, bias_scale=1., FFT=False,
                                         layer_collection=self.layer_collection,
                                         registered=self.registered)
            conv1 = act(conv1)
            conv2 = tf_.circular_conv_2d(conv1, inputShape[1], self.alpha, self.alpha*2, 'conv2',
                                         stride_size=2, biases=True, bias_scale=1., FFT=False,
                                         layer_collection=self.layer_collection,
                                         registered=self.registered)
            conv2 = act(conv2)
            conv3 = tf_.circular_conv_2d(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2, 'conv3',
                                         stride_size=1, biases=True, bias_scale=1., FFT=False,
                                         layer_collection=self.layer_collection,
                                         registered=self.registered)
            conv3 = act(conv3)

            pool3 = tf.reduce_mean(conv3, [1, 2], keepdims=False)
            # pool1 = tf.Print(pool1,[pool1[:3,:], 'pool1'])

            out_re = tf_.fc_layer(pool3, self.alpha*2, 1, 'out_re', biases=False,
                                  layer_collection=self.layer_collection, registered=self.registered)
            out_re = tf.clip_by_value(out_re, -60., 60.)
            # out_re = tf.Print(out_re, [out_re[:3,:], 'out_re'])

            out_im = tf_.fc_layer(pool3, self.alpha*2, 1, 'out_im', biases=False,
                                  layer_collection=self.layer_collection, registered=self.registered)
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            log_prob = tf.complex(out_re, out_im)
            # out = tf.multiply(tf.exp(out_re), tf.sin(out_im))
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        self.registered=True
        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_real_ResNet10_2d(self, x, activation):
        act = tf_.select_activation(activation)
        inputShape = x.get_shape().as_list()
        Lx = int(inputShape[1])
        Ly = int(inputShape[2])
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = tf.cast(x, dtype=self.TF_FLOAT)
            x = tf_.circular_conv_2d(x, 3, inputShape[-1], self.alpha * 64, 'conv1',
                                     stride_size=1, biases=True, bias_scale=1., FFT=False)
            x = tf_.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = act(x)
            for i in range(3):
                x = tf_.residual_block(x, self.alpha * 64, "block_"+str(i),
                                       stride_size=1, activation=act)

            # x = tf_.conv_layer2d(x, 1, self.alpha * 64, 1, "head_conv1")
            x = tf.reduce_mean(x, [1, 2], keepdims=False)
            x = tf_.batch_norm(x, phase=self.bn_is_training, scope='head_bn1')
            x = act(x)

            x = tf.reshape(x, [-1, self.alpha * 64])
            fc1 = tf_.fc_layer(x, self.alpha * 64, self.alpha * 64, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.alpha * 64, 2, 'fc2')
            out = tf.multiply(tf.exp(fc2[:,0]), tf.sin(fc2[:,1]))
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            raise NotImplementedError
            return None,None
        else:
            return out, None

    def build_real_ResNet3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        inputShape = x.get_shape().as_list()
        Lx = int(inputShape[1])
        Ly = int(inputShape[2])
        with tf.variable_scope("network", reuse=None):
            x = tf.cast(x, dtype=self.TF_FLOAT)
            x = tf_.circular_conv_2d(x, 3, inputShape[-1], self.alpha * 64, 'conv1',
                                     stride_size=1, biases=True, bias_scale=1., FFT=False)
            x = tf_.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = act(x)
            for i in range(20):
                x = tf_.residual_block(x, self.alpha * 64, "block_"+str(i),
                                       stride_size=1, activation=act)

            x = tf_.circular_conv_2d(x, 2, self.alpha * 64, 2, 'conv2',
                                     stride_size=2, biases=True, bias_scale=1., FFT=False)
            x = act(x)

            x = tf.reduce_mean(x, [1, 2], keepdims=False)
            out_re = x[:,0]
            # out_re = tf.Print(out_re, [out_re[:3], 'out_re'])
            out_im = x[:,1]
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            log_prob = tf.complex(out_re, out_im)
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_real_ResNet4_2d(self, x, activation):
        act = tf_.select_activation(activation)
        inputShape = x.get_shape().as_list()
        Lx = int(inputShape[1])
        Ly = int(inputShape[2])
        with tf.variable_scope("network", reuse=None):
            x = tf.cast(x, dtype=tf.float32)
            x = tf_.circular_conv_2d(x, 3, inputShape[-1], self.alpha * 64, 'conv1',
                                     stride_size=1, biases=True, bias_scale=1., FFT=False)
            # x = tf_.batch_norm(x, phase=self.bn_is_training, scope='bn1')
            x = act(x)
            for i in range(2):
                x = tf_.residual_block(x, self.alpha * 64, "block_"+str(i),
                                       stride_size=1, activation=act)

            x = tf_.circular_conv_2d(x, 2, self.alpha * 64, 2, 'conv2',
                                     stride_size=2, biases=True, bias_scale=1., FFT=False)
            x = act(x)

            x = tf.reduce_sum(x, [1, 2], keepdims=False)
            out_re = x[:,0]
            # out_re = tf.Print(out_re, [out_re[:3,:], 'out_re'])
            out_im = x[:,1]
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            log_prob = tf.complex(out_re, out_im)
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.exp(log_prob)
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_Jastrow_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, :]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # def jastrow_2d_amp(config_array, Lx, Ly, local_d, name, sym=False):
            out = tf_.jastrow_2d_amp(x, inputShape[1], inputShape[2], inputShape[-1], 'jastrow')
            out = tf.real((out))

        if self.using_complex:
            raise NotImplementedError
        else:
            return out, None

    def build_network_1d(self, which_net, x, activation, num_blocks):
        if which_net == "NN":
            return self.build_NN_1d(x)
        elif which_net == "ZNet":
            return self.build_ZNet_1d(x)
        elif which_net == "NN3":
            return self.build_NN3_1d(x)
        elif which_net == "CNN":
            return self.build_CNN_1d(x)
        elif which_net == "FCN1":
            return self.build_FCN1_1d(x)
        elif which_net == "FCN2":
            return self.build_FCN2_1d(x)
        elif which_net == "FCN3":
            return self.build_FCN3_1d(x)
        elif which_net == "NN_complex":
            return self.build_NN_complex(x, activation)
        elif which_net == "NN3_complex":
            return self.build_NN3_complex(x, activation)
        elif which_net == "RBM":
            return self.build_RBM_1d(x)
        elif which_net == "RBM_cosh":
            return self.build_RBM_cosh_1d(x)
        elif which_net == "sRBM":
            return self.build_sRBM_1d(x)
        elif which_net == "ResNet":
            return self.build_ResNet(x, activation)
        else:
            raise NotImplementedError

    def build_network_2d(self, which_net, x, activation, num_blocks):
        if which_net == "NN":
            return self.build_NN_2d(x, activation)
        if which_net == "NN_linear":
            return self.build_NN_linear_2d(x, activation)
        elif which_net == "NN3":
            return self.build_NN3_2d(x, activation)
        elif which_net == 'MADE':
            return self.build_MADE_2d(x, activation)
        elif which_net == 'pixelCNN':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          mode='2',
                                         )
        elif which_net == 'pixelCNNv2':
            return self.build_pixelCNNv2_2d(x,
                                            activation,
                                            num_blocks,
                                           )
        elif which_net == 'pixelCNN-BN':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          mode='2',
                                          BN=True,
                                         )
        elif which_net == 'pixelCNN-Res':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          residual_connection=True,
                                          mode='2',
                                         )
        elif which_net == 'pixelCNN-Res-BN':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          residual_connection=True,
                                          mode='2',
                                          BN=True,
                                         )
        elif which_net == 'pixelCNN-WN':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          mode='2',
                                          weight_normalization=True,
                                         )
        elif which_net == 'pixelCNN-Res-WN':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          residual_connection=True,
                                          mode='2',
                                          weight_normalization=True,
                                         )
        elif which_net == 'pixelCNN-Agg':
            return self.build_pixelCNN_2d(x,
                                          activation,
                                          num_blocks,
                                          residual_connection=False,
                                          mode='2',
                                          BN=False,
                                          split_block=True)
        elif which_net == "ResNN3":
            return self.build_ResNN3_2d(x, activation)
        elif which_net == "RBM":
            return self.build_RBM_2d(x)
        elif which_net == "RBM_cosh":
            return self.build_RBM_cosh_2d(x)
        elif which_net == "sRBM":
            return self.build_sRBM_2d(x, activation)
        elif which_net == "FCN2v0":
            return self.build_FCN2v0_2d(x, activation)
        elif which_net == "real_FCN2":
            return self.build_real_FCN2_2d(x, activation)
        elif which_net == "FCN2":
            return self.build_FCN2_2d(x, activation)
        elif which_net == "FCN3v1":
            return self.build_FCN3v1_2d(x, activation)
        elif which_net == "FCN3v2":
            return self.build_FCN3v2_2d(x, activation)
        elif which_net == "real_CNN":
            return self.build_real_CNN_2d(x, activation)
        elif which_net == "real_CNN3":
            return self.build_real_CNN3_2d(x, activation)
        elif which_net == "Jastrow":
            return self.build_Jastrow_2d(x)
        elif which_net == "pre_sRBM":
            return self.build_pre_sRBM_2d(x)
        elif which_net == "real_ResNet10":
            return self.build_real_ResNet10_2d(x, activation)
        elif which_net == "real_ResNet3":
            return self.build_real_ResNet3_2d(x, activation)
        elif which_net == "real_ResNet4":
            return self.build_real_ResNet4_2d(x, activation)
        else:
            raise NotImplementedError
