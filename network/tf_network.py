import numpy as np
import math
import tensorflow as tf
from .hoshen_kopelman import  label
from . import tf_wrapper as tf_


class tf_network:
    def __init__(self, which_net, inputShape, optimizer, dim,
                 learning_rate=0.1125, momentum=0.90, alpha=2,
                 activation=None, using_complex=True, single_precision=True):
        '''
        using_complex:
            using complex-valued/real-valued wavefunction or not.
            This would affect the type of other variables, including
            local energy.
        '''
        ##################
        # Parameters
        ##################
        self.learning_rate = tf.Variable(learning_rate, name='learning_rate')
        self.momentum = tf.Variable(momentum, name='momentum')
        self.exp_stabilizer = tf.Variable(0., name="exp_stabilizer")
        # dropout = 0.75  # Dropout, probability to keep units
        self.bn_is_training = True
        self.max_bp_batch_size = 512
        self.max_fp_batch_size = 5120
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

        ##########################
        # tf Graph input & Create network
        ##########################
        self.alpha = alpha
        self.activation = activation
        self.which_net = which_net
        self.using_complex = using_complex
        self.keep_prob = tf.placeholder(self.TF_FLOAT)
        self.dx_exp_stabilizer = tf.placeholder(self.TF_FLOAT)
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
            self.LxLy = self.Lx * self.Ly
            self.channels = int(inputShape[2])
            if self.channels > 2:
                print("Not yet implemented for tJ, Hubbard model")
                raise NotImplementedError
            else:
                pass


            self.build_network = self.build_network_2d

        else:
            raise NotImplementedError

        #########################
        # Variables Creation
        #########################
        self.pred, self.log_psi = self.build_network(which_net, self.x, self.activation)
        if self.log_psi is None:
            # self.log_psi = tf.log(self.pred)
            # For real-valued wavefunction, log_psi is only a intermediate step for
            # Log gradient. log_psi should not be read out. Otherwise, one need to
            # define as below,
            #
            self.log_psi = tf.log(tf.cast(self.pred, self.TF_COMPLEX))
            # Cast type to complex before log, to prevent nan in
            # the case for tf.float input < 0
            #
            # This is not a problem if one do not want to read out
            # The log value explicitly, but only need the derivative of log.
            #
            # But to prevent error, always cast to TF_COMPLEX.

        self.model_var_list = tf.global_variables()
        self.para_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        print("create variable")
        self.para_list_wo_bn=[]
        for i in self.para_list:
            print(i.name)
            if 'bn' in i.name:
                pass
            else:
                self.para_list_wo_bn.append(i)

        self.para_list = self.para_list_wo_bn
        # para_list are list of variables for gradient calculation
        # could not include batchnorm variables.

        self.var_shape_list = [var.get_shape().as_list() for var in self.para_list]
        self.num_para = self.getNumPara()
        # Define optimizer
        self.optimizer = tf_.select_optimizer(optimizer, self.learning_rate,
                                              self.momentum)

        # Below we define the gradient.
        # tf.gradient(cost, variable_list)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Define Log Gradient, loss = log(wave function)
            # Define Energy Gradient, loss = E(wave function)
            if not using_complex:
                # (1.1) real log_grads
                # real-valued wavefunction, log_psi, grad_log_psi are all real
                self.log_grads = tf.gradients(self.log_psi, self.para_list, grad_ys=tf.complex(1.,0.))
                # (2.1)
                self.E_grads = tf.gradients(self.log_psi, self.para_list, grad_ys=tf.complex(self.E_loc_m_avg,0.))
                # log_psi is always complex, so we need to specify grad_ys
            else:
                # (1.2) complex log_grads
                # complex-valued wavefunction, log_psi are all complex, but
                # grad_log_psi would be cast to real by tensorflow default !
                # To prevent this, we manually compute the gradient.
                # Cast gradient back to real only before apply_gradient.
                self.log_grads_real = tf.gradients(tf.real(self.log_psi), self.para_list)
                self.log_grads_imag = tf.gradients(tf.imag(self.log_psi), self.para_list)
                self.log_grads = [tf.complex(self.log_grads_real[i], self.log_grads_imag[i])
                                  for i in range(len(self.log_grads_real))]
                # (2.1)
                self.E_grads = tf.gradients(self.log_psi, self.para_list, grad_ys=self.E_loc_m_avg)
                # log_psi is always complex, so we need to specify grad_ys


        # Pseudo Code for batch Gradient
        # examples = tf.split(self.x)
        # weight_copies = [tf.identity(self.para_list) for x in examples]
        # output = tf.stack(f(x, w) for x, w in zip(examples, weight_copies))
        # cost = tf.log(output)
        # per_example_gradients = tf.gradients(cost, weight_copies)

        # Unaggregated Gradient with while_loop
        self.unaggregated_gradient = self.build_unaggregated_gradient()

        # Do some operation on grads
        # Get the new gradient from outside by placeholder
        self.newgrads = [tf.placeholder(self.TF_FLOAT, g.get_shape()) for g in self.log_grads]
        self.train_op = self.optimizer.apply_gradients(zip(self.newgrads,
                                                           self.para_list))

        self.update_exp_stabilizer = self.exp_stabilizer.assign(self.exp_stabilizer +
                                                                self.dx_exp_stabilizer)

        if which_net in ['pre_sRBM']:
            self.forwardPass = self.pre_forwardPass
            self.backProp = self.pre_backProp
            self.vanilla_back_prop = self.pre_vanilla_back_prop
        else:
            self.forwardPass = self.plain_forwardPass
            self.forwardPass_log_psi = self.plain_forwardPass_log_psi
            self.backProp = self.plain_backProp
            self.vanilla_back_prop = self.plain_vanilla_back_prop


        # Initializing All the variables and operation, all operation and variables should 
        # be defined before here.!!!!
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)
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

    def plain_forwardPass(self, X0):
        return self.sess.run(self.pred, feed_dict={self.x: X0, self.keep_prob: 1.})

    def plain_forwardPass_log_psi(self, X0):
        return self.sess.run(self.log_psi, feed_dict={self.x: X0, self.keep_prob: 1.})

    def plain_backProp(self, X0):
        return self.sess.run(self.log_grads, feed_dict={self.x: X0, self.keep_prob: 1.})

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
        else:
            unaggregated_grad = tf.TensorArray(dtype=self.TF_FLOAT, size=tf.shape(self.x)[0])

        init_state = (0, unaggregated_grad)
        # i = tf.constant(0)
        condition = lambda i, _: i < tf.shape(self.x)[0]
        def body(i, ta):
            single_x = self.x[i:i+1]
            if self.using_complex:
                single_log_psi = tf.log(self.build_network(self.which_net, single_x, self.activation)[0])
                single_log_grads_real = tf.gradients(tf.real(single_log_psi), self.para_list)
                single_log_grads_imag = tf.gradients(tf.imag(single_log_psi), self.para_list)
                single_log_grads = [tf.complex(single_log_grads_real[j], single_log_grads_imag[j])
                                    for j in range(len(single_log_grads_real))]
                ta = ta.write(i, tf.concat([tf.reshape(g,[-1]) for g in single_log_grads], axis=0 ))
            else:
                single_log_psi = tf.log(tf.cast(self.build_network(self.which_net, single_x, self.activation)[0], self.TF_COMPLEX))
                ta = ta.write(i, tf.concat([tf.reshape(g,[-1]) for g in tf.gradients(single_log_psi, self.para_list, grad_ys=tf.complex(1.,0.))], axis=0 ))
            return (i+1, ta)

        n, final_unaggregated_grad = tf.while_loop(condition, body, init_state, back_prop=False)
        final_unaggregated_grad = tf.transpose(final_unaggregated_grad.stack())
        return final_unaggregated_grad

    def run_unaggregated_gradient(self, X0):
        return self.sess.run(self.unaggregated_gradient, feed_dict={self.x: X0})

    def plain_vanilla_back_prop(self, X0, E_loc_array):
        # Implementation below fail for unknown reason
        # Not sure whether it is bug from tensorflow or not.
        # 
        # E_vec = (self.E_loc - tf.reduce_mean(self.E_loc))
        # E = tf.reduce_sum(tf.multiply(E_vec, log_psi))
        # E = (tf.multiply(E_vec, log_psi))
        # return self.sess.run(tf.gradients(E, self.para_list),

        # because grad_ys has to have the same shape as ys
        # we need to reshape E_loc_array as [None, 1]
        '''
        Input (numpy array):
            X0 : the config array
            E_loc_array : E_array - E_avg
        Output:
            returning the gradient, in python list of numpy array.

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
                                                  self.E_loc_m_avg: E_loc_array[max_bp_size*idx : max_bp_size*(idx+1)]})
                grad_array += np.concatenate([g.flatten() for g in G_list])

            G_list = self.sess.run(self.E_grads,
                                   feed_dict={self.x: X0[max_bp_size*(num_data//max_bp_size):],
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
                                 feed_dict={self.x: X0, self.E_loc_m_avg: E_loc_array})

    def pre_forwardPass(self, X0):
        X0 = self.enrich_features(X0)
        return self.sess.run(self.pred, feed_dict={self.x: X0, self.keep_prob: 1.})

    def pre_backProp(self, X0):
        X0 = self.enrich_features(X0)
        return self.sess.run(self.log_grads, feed_dict={self.x: X0, self.keep_prob: 1.})

    def pre_vanilla_back_prop(self, X0, E_loc_array):
        X0 = self.enrich_features(X0)
        return self.plain_vanilla_back_prop(X0, E_loc_array)

    def getNumPara(self):
        for i in self.para_list:
            print(i.name, i.get_shape().as_list())

        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    def exp_stabilizer_add(self, increments):
        self.sess.run(self.update_exp_stabilizer, feed_dict={self.dx_exp_stabilizer: increments})

    def build_NN_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
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
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'fc1')
            fc1 = act(fc1)
            out_re = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_re')
            out_re = tf.clip_by_value(out_re, -60., 60.)
            out_im = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_im')
            log_psi = tf.copmlex(out_re, out_im)
            out = tf.exp(log_psi)

        if self.using_complex:
            return out, log_psi
        else:
            return tf.real(out), None

    def build_NN_linear_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
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
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2')
            fc2 = act(fc2)
            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3')
            fc3 = tf.nn.tanh(fc3)
            out_re = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_im')
            log_psi = tf.complex(out_re, out_im)
            out = tf.exp(log_psi)

        if self.using_complex:
            return out, log_psi
        else:
            return tf.real(out), None

    def build_NN3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0]
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha//2, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc2')
            fc2 = act(fc2)
            fc3 = tf_.fc_layer(fc2, self.LxLy * self.alpha //2, self.LxLy * self.alpha //2, 'fc3')
            fc3 = act(fc3)
            out_re = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_re')
            out_re = tf.clip_by_value(out_re, -60., 60.)
            out_im = tf_.fc_layer(fc3, self.LxLy * self.alpha //2, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out, tf.complex(out_re, out_im)

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
#             out = tf.reduce_sum(pool4, [1], keep_dims=True)
#             out_re = tf.real(out)  # + tf_.fc_layer(x[:, :, 0], self.L, 1, 'v_bias_re')
#             out_im = tf.imag(out)
#
#             out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
#
#         return out

    def build_CNN_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2], keep_dims=False)
            pool4 = tf.exp(pool4)

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keep_dims=False)

            # Fully connected layer
            # fc_dim = self.alpha  # np.prod(pool4.get_shape().as_list()[1:])
            # pool4 = tf.reshape(pool4, [-1, fc_dim])
            # out = tf_.fc_layer(pool4, fc_dim, 1, 'out', biases=False, dtype=tf.complex64)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keep_dims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])
            # out = tf.reshape(pool4, [-1, 1])

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
            # sym_phase = tf.reduce_sum(sym_phase, [1], keep_dims=True)
            # print(sym_phase.get_shape().as_list())
            # sym_phase = tf.real(tf.log(tf.complex(sym_phase + 1e-8, 0.)))
            # print(out_im.get_shape().as_list())
            # out_im = tf.add(out_im, sym_phase)
            # print(out_im.get_shape().as_list())

            # out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
            # out = out * tf.exp(tf.complex(0., tf.Variable([1], 1.0, dtype=self.TF_FLOAT)))
            out = tf.real((out))
            return out, None

    def build_FCN1_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300.)
            conv1 = tf_.complex_relu_neg(tf.complex(conv1_re, conv1_im))
            conv2 = conv1

            pool4 = tf.reduce_sum(conv2, [1, 2], keep_dims=False)
            pool4 = tf.exp(pool4)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100.)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100.)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keep_dims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])
            out = tf.real((out))
            return out, None

    def build_FCN2_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1]/2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1]/2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=300.)
            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            conv2 = tf_.circular_conv_1d_complex(conv1, inputShape[1]/2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=100.)
            conv2 = tf_.softplus2(conv2)

            log_psi = tf.reduce_sum(conv2, [1, 2], keep_dims=False)
            pool3 = tf.exp(log_psi)

            ## Conv Bias
            # conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
            #                           [1, 2], keep_dims=False)
            # out = tf.reshape(tf.multiply(pool3, tf.exp(conv_bias)), [-1, 1])
            out = tf.reshape(pool3, [-1, 1])
            log_psi = tf.reshape(log_psi, [-1, 1])
            out = tf.real((out))
            return out, log_psi

    def build_FCN3_1d(self, x):
        act = tf_.softplus2
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1]/4, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1]/4, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300.)
            conv1 = act(tf.complex(conv1_re, conv1_im))
            conv2 = tf_.circular_conv_1d_complex(conv1, inputShape[1]/4, self.alpha, self.alpha,
                                                 'conv2_complex', stride_size=1, biases=True,
                                                 bias_scale=100.)
            conv2 = act(conv2)
            conv3 = tf_.circular_conv_1d_complex(conv2, inputShape[1]/4, self.alpha, self.alpha,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=100.)
            conv3 = act(conv3)

            ## Pooling
            pool4 = tf.reduce_sum(conv3, [1, 2], keep_dims=False)
            log_psi = tf.reshape(pool4, [-1, 1])
            out = tf.exp(log_psi)

            ## FC layer
            # conv3 = tf.reduce_sum(conv3, [1], keep_dims=False)
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
            #                           [1, 2], keep_dims=False)
            # conv_bias = tf.exp(conv_bias)
            # out = tf.reshape(tf.multiply(pool4, conv_bias), [-1, 1])

            out = tf.real((out))
            return out, log_psi

    def build_ResNet(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = tf.nn.softplus(fc1)
            fc1 = fc1 + x

            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2')
            fc2 = tf.nn.softplus(fc2)
            fc2 = fc2 + fc1

            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3')
            fc3 = tf.nn.softplus(fc3)

            out_re = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_re')
            out_re = out_re + tf_.fc_layer(x, self.L, 1, 'v_bias')
            out_im = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out, tf.complex(out_re, out_im)

    def build_RBM_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            # inputShape = x.get_shape().as_list()
            x = tf.cast(x[:, :, 0], dtype=self.TF_FLOAT)
            fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.softplus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias_re = tf_.fc_layer(x, self.L, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, self.L, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out, log_prob

    def build_RBM_cosh_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            # inputShape = x.get_shape().as_list()
            x = x[:, :, 0]
            fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)

            v_bias_re = tf_.fc_layer(x, self.L, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, self.L, 1, 'v_bias_im')

            # !!! The implementation below fail for calculting gradient !!!
            # !!! forward prediction is correct !!!
            # fc2 = tf.cosh(fc1)
            # v_bias = tf.exp(tf.complex(v_bias_re, v_bias_im))
            # out = tf.multiply(v_bias, tf.reduce_prod(fc2, axis=1, keep_dims=True))
            # out = tf.real(out)
            fc2 = tf.log(tf.cosh(fc1))
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out, log_prob


    def build_sRBM_1d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.softplus2(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2], keep_dims=False)
            pool4 = tf.exp(pool4)

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keep_dims=False)

            conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2], keep_dims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])
            out = tf.real((out))
            return out, None

    def build_NN_complex(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1_complex',
                                       dtype=tf.complex64)
            fc1_complex = tf_.softplus(fc1_complex)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, 1, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)

            out = tf.exp(fc2_complex)
            out = tf.real(out)

        return out, fc2_complex

    def build_NN3_complex(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, 0]
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1_complex',
                                       dtype=tf.complex64)
            fc1_complex = tf_.softplus(fc1_complex)
            # fc1_complex = fc1_complex + tf.complex(x, 0.)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, self.L * self.alpha, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)
            fc2_complex = tf_.softplus(fc2_complex)
            # fc2_complex = fc2_complex + fc1_complex

            fc3_complex = tf_.fc_layer(fc2_complex, self.L * self.alpha, self.L * self.alpha, 'fc3_complex',
                                       dtype=tf.complex64, biases=True)
            fc3_complex = tf_.softplus(fc3_complex)
            # fc3_complex = fc3_complex  + fc2_complex

            # fc4_complex = tf.reduce_sum(fc3_complex, axis=1, keep_dims=True)
            # out = tf.multiply(tf.exp(tf.real(fc4_complex) / self.L), tf.cos(tf.imag(fc4_complex)))
            fc4_complex = tf_.fc_layer(fc3_complex, self.L * self.alpha, 1, 'fc4_complex',
                                       dtype=tf.complex64, biases=True)
            out = tf.exp(fc4_complex)
            out = tf.real(out)

        return out, fc4_complex

    def build_RBM_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            inputShape = x.get_shape().as_list()
            b_size, Lx, Ly, _ = inputShape
            LxLy = Lx * Ly
            x = tf.reshape(x[:, :, :, 0], [-1, LxLy])
            x = tf.cast(x, dtype=self.TF_FLOAT)
            fc1_re = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.softplus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out, log_prob

    def build_RBM_cosh_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            inputShape = x.get_shape().as_list()
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
            # out = tf.multiply(v_bias, tf.reduce_prod(fc2, axis=1, keep_dims=True))
            # out = tf.real(out)

            fc2 = tf.log(tf.cosh(fc1))
            v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out, log_prob

    def build_sRBM_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, dtype=self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False)
                                            # stride_size=2, biases=True, bias_scale=3000.*2/64/self.alpha, FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=1, FFT=False)
                                            # stride_size=2, biases=True, bias_scale=3140.*2/64/self.alpha, FFT=False)

            conv1 = act(tf.complex(conv1_re, conv1_im))
            # conv1 = tf_.complex_relu(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2, 3], keep_dims=False)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = tf.real(pool4)
            pool4_imag = tf.imag(pool4)
            # pool4 = tf.exp(tf.complex(pool4_real, pool4_imag))
            # pool4 = tf.exp(pool4)
            # out = tf.real((out))
            # return tf.reshape(tf.real(pool4), [-1,1])

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keep_dims=False)

            conv_bias_re = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias_im = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2, 3], keep_dims=False)
            final_real = tf.clip_by_value(pool4_real + tf.real(conv_bias), -60., 60.)
            final_imag = pool4_imag + tf.imag(conv_bias)
            out = tf.reshape(tf.exp(tf.complex(final_real, final_imag)), [-1, 1])

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
            pool4 = tf.reduce_sum(conv1, [1, 2, 3], keep_dims=False)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = tf.real(pool4)
            pool4_imag = tf.imag(pool4)
            # pool4 = tf.exp(tf.complex(pool4_real, pool4_imag))
            # pool4 = tf.exp(pool4)
            # out = tf.real((out))
            # return tf.reshape(tf.real(pool4), [-1,1])

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keep_dims=False)

            conv_bias_re = tf_.circular_conv_2d(x[:, :, :, 0:2], 2, 2, 1, 'conv_bias_re',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias_im = tf_.circular_conv_2d(x[:, :, :, 0:2], 2, 2, 1, 'conv_bias_im',
                                                stride_size=2, biases=False, bias_scale=1., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2, 3], keep_dims=False)
            final_real = pool4_real + tf.real(conv_bias)
            # final_real = tf.clip_by_value(final_real, -60., 60.)
            final_real = final_real - self.exp_stabilizer
            final_imag = pool4_imag + tf.imag(conv_bias)
            log_prob = tf.reshape(tf.complex(final_real, final_imag), [-1, 1])
            out = tf.exp(log_prob)
            out = tf.real((out))

            return out, log_prob




    def build_FCN2_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            x = tf.cast(x, self.TF_FLOAT)
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1.)
            conv2 = act(conv2)

            pool3 = tf.reduce_sum(conv2[:, :, :, :self.alpha], [1, 2, 3], keep_dims=False) -\
                    tf.reduce_sum(conv2[:, :, :, self.alpha:], [1, 2, 3], keep_dims=False)

            # pool3_real = tf.real(pool3)
            pool3_real = tf.clip_by_value(tf.real(pool3), -70., 70.)
            # pool3_real = tf.real(pool3) - self.exp_stabilizer
            pool3_imag = tf.imag(pool3)
            log_prob = tf.complex(pool3_real,pool3_imag)
            out = tf.exp(log_prob)
            out = tf.reshape(tf.real(out), [-1, 1])

        if self.using_complex:
            return out, log_prob
        else:
            return tf.real(out), None

    def build_FCN2v1_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1.)
            conv2 = act(conv2)

            pool3 = tf.reduce_sum(conv2, [1, 2, 3], keep_dims=False)
            pool3_real = tf.clip_by_value(tf.real(pool3), -60., 60.)
            pool3_imag = tf.imag(pool3)
            log_prob = tf.complex(pool3_real,pool3_imag)
            out = tf.exp(log_prob)
            out = tf.real((out))

            out = tf.reshape(out, [-1, 1])
        return out, log_prob

    def build_FCN3v1_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1.)
            conv2 = act(conv2)

            conv3 = tf_.circular_conv_2d_complex(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=1.)
            conv3 = act(conv3)

            # pool4 = tf.reduce_sum(conv3[:, :, :, :self.alpha], [1, 2, 3], keep_dims=False) -\
            #         tf.reduce_sum(conv3[:, :, :, self.alpha:], [1, 2, 3], keep_dims=False)
            pool4 = tf.reduce_sum(conv3, [1, 2, 3], keep_dims=False)
            pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_imag = tf.imag(pool4)
            log_prob = tf.complex(pool4_real, pool4_imag)
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.exp(log_prob)

            out = tf.reshape(out, [-1, 1])
            out = tf.real((out))
        return out, log_prob

    def build_FCN3v2_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=3., FFT=False)
            conv1 = act(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1]//2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=1.)
            conv2 = act(conv2)

            conv3 = tf_.circular_conv_2d_complex(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2,
                                                 'conv3_complex', stride_size=1, biases=True,
                                                 bias_scale=1.)
            conv3 = act(conv3)

            pool4 = tf.reduce_sum(conv3, [1, 2, 3], keep_dims=False)
            pool4_real = tf.real(pool4)
            # pool4_real = tf.clip_by_value(tf.real(pool4), -60., 60.)
            pool4_real = pool4_real - self.exp_stabilizer
            pool4_imag = tf.imag(pool4)
            log_prob = tf.complex(pool4_real, pool4_imag)
            out = tf.exp(log_prob)

            out = tf.reshape(out, [-1, 1])
            log_prob = tf.reshape(log_prob, [-1, 1])
            out = tf.real((out))
        return out, log_prob

    def build_real_CNN_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1 = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1',
                                         stride_size=2, biases=True, bias_scale=1., FFT=False)
            conv1 = act(conv1)
            pool1 = tf.reduce_mean(conv1, [1, 2], keep_dims=False)
            # pool1 = tf.Print(pool1,[pool1[:3,:], 'pool1'])

            out_re = tf_.fc_layer(pool1, self.alpha, 1, 'out_re', biases=False)
            out_re = tf.clip_by_value(out_re, -60., 60.)
            # out_re = tf.Print(out_re, [out_re[:3,:], 'out_re'])

            out_im = tf_.fc_layer(pool1, self.alpha, 1, 'out_im', biases=False)
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            out = tf.multiply(tf.exp(out_re), tf.sin(out_im))
            out = tf.reshape(out, [-1, 1])

        return out, tf.complex(out_re, math.pi/2.-out_im)

    def build_real_CNN3_2d(self, x, activation):
        act = tf_.select_activation(activation)
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1 = tf_.circular_conv_2d(x, inputShape[1]//2, inputShape[-1], self.alpha, 'conv1',
                                         stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv1 = act(conv1)
            conv2 = tf_.circular_conv_2d(conv1, inputShape[1]//2, self.alpha, self.alpha*2, 'conv2',
                                         stride_size=2, biases=True, bias_scale=1., FFT=False)
            conv2 = act(conv2)
            conv3 = tf_.circular_conv_2d(conv2, inputShape[1]//2, self.alpha*2, self.alpha*2, 'conv3',
                                         stride_size=1, biases=True, bias_scale=1., FFT=False)
            conv3 = act(conv3)

            pool3 = tf.reduce_mean(conv3, [1, 2], keep_dims=False)
            # pool1 = tf.Print(pool1,[pool1[:3,:], 'pool1'])

            out_re = tf_.fc_layer(pool3, self.alpha*2, 1, 'out_re', biases=False)
            out_re = tf.clip_by_value(out_re, -60., 60.)
            # out_re = tf.Print(out_re, [out_re[:3,:], 'out_re'])

            out_im = tf_.fc_layer(pool3, self.alpha*2, 1, 'out_im', biases=False)
            # out_im = tf.Print(out_im, [out_im[:3,:], 'out_im'])
            out = tf.multiply(tf.exp(out_re), tf.sin(out_im))
            out = tf.reshape(out, [-1, 1])
        return out, tf.complex(out_re, math.pi/2.-out_im)

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
            for i in range(10):
                x = tf_.residual_block(x, self.alpha * 64, "block_"+str(i),
                                       stride_size=1, activation=act)

            x = tf_.conv_layer2d(x, 1, self.alpha * 64, 1, "head_conv1")
            x = tf_.batch_norm(x, phase=self.bn_is_training, scope='head_bn1')
            x = act(x)

            x = tf.reshape(x, [-1, Lx*Ly])
            fc1 = tf_.fc_layer(x, Lx*Ly, self.alpha * 64, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.alpha * 64, 2, 'fc2')
            out = tf.multiply(tf.exp(fc2[:,0]), tf.sin(fc2[:,1]))
            out = tf.reshape(out, [-1, 1])

        if self.using_complex:
            return None,None
        else:
            return out, None

    def build_real_ResNet20_2d(self, x, activation):
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
            for i in range(20):
                x = tf_.residual_block(x, self.alpha * 64, "block_"+str(i),
                                       stride_size=1, activation=act)

            x = tf_.conv_layer2d(x, 1, self.alpha * 64, 1, "head_conv1")
            x = tf_.batch_norm(x, phase=self.bn_is_training, scope='head_bn1')
            x = act(x)

            x = tf.reshape(x, [-1, Lx*Ly])
            fc1 = tf_.fc_layer(x, Lx*Ly, self.alpha * 64, 'fc1')
            fc1 = act(fc1)
            fc2 = tf_.fc_layer(fc1, self.alpha * 64, 2, 'fc2')
            out = tf.multiply(tf.exp(fc2[:,0]), tf.sin(fc2[:,1]))
            out = tf.reshape(out, [-1, 1])

        return out, tf.complex(fc2[:,0], math.pi/2. - fc2[:,1])

    def build_Jastrow_2d(self, x):
        with tf.variable_scope("network", reuse=tf.AUTO_REUSE):
            x = x[:, :, :, :]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # def jastrow_2d_amp(config_array, Lx, Ly, local_d, name, sym=False):
            out = tf_.jastrow_2d_amp(x, inputShape[1], inputShape[2], inputShape[-1], 'jastrow')
            out = tf.real((out))
        return out, None

    def build_network_1d(self, which_net, x, activation):
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
            return self.build_NN_complex(x)
        elif which_net == "NN3_complex":
            return self.build_NN3_complex(x)
        elif which_net == "RBM":
            return self.build_RBM_1d(x)
        elif which_net == "RBM_cosh":
            return self.build_RBM_cosh_1d(x)
        elif which_net == "sRBM":
            return self.build_sRBM_1d(x)
        elif which_net == "ResNet":
            return self.build_ResNet(x)
        else:
            raise NotImplementedError

    def build_network_2d(self, which_net, x, activation):
        if which_net == "NN":
            return self.build_NN_2d(x, activation)
        if which_net == "NN_linear":
            return self.build_NN_linear_2d(x, activation)
        elif which_net == "NN3":
            return self.build_NN3_2d(x, activation)
        elif which_net == "RBM":
            return self.build_RBM_2d(x)
        elif which_net == "RBM_cosh":
            return self.build_RBM_cosh_2d(x)
        elif which_net == "sRBM":
            return self.build_sRBM_2d(x, activation)
        elif which_net == "FCN2":
            return self.build_FCN2_2d(x, activation)
        elif which_net == "FCN2v1":
            return self.build_FCN2v1_2d(x, activation)
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
        elif which_net == "real_ResNet20":
            return self.build_real_ResNet20_2d(x, activation)
        else:
            raise NotImplementedError
