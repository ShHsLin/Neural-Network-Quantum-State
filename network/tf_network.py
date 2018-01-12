import numpy as np
import tensorflow as tf
from . import tf_wrapper as tf_


class tf_network:
    def __init__(self, which_net, inputShape, optimizer, dim,
                 learning_rate=0.1125, momentum=0.90, alpha=2):
        # Parameters
        self.learning_rate = tf.Variable(learning_rate)
        self.momentum = tf.Variable(momentum)
        # dropout = 0.75  # Dropout, probability to keep units

        # tf Graph input
        self.alpha = alpha
        self.keep_prob = tf.placeholder(tf.float32)
        if dim == 1:
            self.x = tf.placeholder(tf.float32, [None, inputShape[0], inputShape[1]])
            self.L = int(inputShape[0])
            self.build_network = self.build_network_1d
        elif dim == 2:
            self.x = tf.placeholder(tf.float32, [None, inputShape[0], inputShape[1], inputShape[2]])
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

        # Variables Creation
        self.pred = self.build_network(which_net, self.x)
        self.model_var_list = tf.global_variables()
        self.para_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        print("create variable")
        for i in self.para_list:
            print(i.name)
        # Define optimizer
        self.optimizer = tf_.select_optimizer(optimizer, self.learning_rate,
                                              self.momentum)

        # Define Gradient, loss = log(wave function)
        self.grads = tf.gradients(tf.log(self.pred), self.para_list)  # grad(cost, variable_list)
        # Pseudo Code for batch Gradient
        # examples = tf.split(self.x)
        # weight_copies = [tf.identity(self.para_list) for x in examples]
        # output = tf.stack(f(x, w) for x, w in zip(examples, weight_copies))
        # cost = tf.log(output)
        # per_example_gradients = tf.gradients(cost, weight_copies)

        # Do some operation on grads
        # Get the new gradient from outside by placeholder
        self.newgrads = [tf.placeholder(tf.float32, g.get_shape()) for g in self.grads]
        self.train_op = self.optimizer.apply_gradients(zip(self.newgrads,
                                                           self.para_list))

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def forwardPass(self, X0):
        return self.sess.run(self.pred, feed_dict={self.x: X0, self.keep_prob: 1.})

    def backProp(self, X0):
        return self.sess.run(self.grads, feed_dict={self.x: X0, self.keep_prob: 1.})

    def getNumPara(self):
        for i in self.para_list:
            print(i.name, i.get_shape().as_list())

        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    def build_NN_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = tf.cos(fc1)
            out_re = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out

    def build_ZNet_1d(self, x):
        with tf.variable_scope("network", reuse=None):
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

        return out



    def build_NN_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, :, 0]
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'fc1')
            fc1 = tf.nn.tanh(fc1)
            out_re = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc1, self.LxLy * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out

    def build_NN3_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = tf.nn.tanh(fc1)
            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2')
            fc2 = tf.nn.tanh(fc2)
            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3')
            fc3 = tf.nn.tanh(fc3)
            out_re = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out

    def build_NN3_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, :, 0]
            fc1 = tf_.fc_layer(x, self.LxLy, self.LxLy * self.alpha, 'fc1')
            fc1 = tf.nn.tanh(fc1)
            fc2 = tf_.fc_layer(fc1, self.LxLy * self.alpha, self.LxLy * self.alpha, 'fc2')
            fc2 = tf.nn.tanh(fc2)
            fc3 = tf_.fc_layer(fc2, self.LxLy * self.alpha, self.LxLy * self.alpha, 'fc3')
            fc3 = tf.nn.tanh(fc3)
            out_re = tf_.fc_layer(fc3, self.LxLy * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc3, self.LxLy * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out

#     def build_CNN_1d(self, x):
#         with tf.variable_scope("network", reuse=None):
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
#             pool4 = tf_.soft_plus(pool4)
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
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.soft_plus2(tf.complex(conv1_re, conv1_im))
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
            #                        'sym_bias', tf.float32)

            # sym_bias = tf.ones([inputShape[1]], tf.float32)
            # sym_bias_fft = tf.fft(tf.complex(sym_bias, 0.))
            # x_fft = tf.fft(tf.complex(x[:, :, 0], 0.))
            # sym_phase = tf.real(tf.ifft(x_fft * tf.conj(sym_bias_fft)))
            # theta = tf.scalar_mul(tf.constant(np.pi),
            #                       tf.range(inputShape[1], dtype=tf.float32))
            # sym_phase = sym_phase * tf.cos(theta)
            # print(sym_phase.get_shape().as_list())
            # sym_phase = tf.reduce_sum(sym_phase, [1], keep_dims=True)
            # print(sym_phase.get_shape().as_list())
            # sym_phase = tf.real(tf.log(tf.complex(sym_phase + 1e-8, 0.)))
            # print(out_im.get_shape().as_list())
            # out_im = tf.add(out_im, sym_phase)
            # print(out_im.get_shape().as_list())

            # out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
            # out = out * tf.exp(tf.complex(0., tf.Variable([1], 1.0, dtype=tf.float32)))
            out = tf.real((out))
            return out

    def build_FCN1_1d(self, x):
        with tf.variable_scope("network", reuse=None):
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
            return out

    def build_FCN2_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1]/2, inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=100.)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1]/2, inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=300.)
            conv1 = tf_.soft_plus2(tf.complex(conv1_re, conv1_im))
            conv2 = tf_.circular_conv_1d_complex(conv1, inputShape[1]/2, self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=100.)
            conv2 = tf_.soft_plus2(conv2)

            pool3 = tf.reduce_sum(conv2, [1, 2], keep_dims=False)
            pool3 = tf.exp(pool3)

            ## Conv Bias
            # conv_bias_re = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_re',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias_im = tf_.circular_conv_1d(x, 2, inputShape[-1], 1, 'conv_bias_im',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
            #                           [1, 2], keep_dims=False)
            # out = tf.reshape(tf.multiply(pool3, tf.exp(conv_bias)), [-1, 1])
            out = tf.reshape(pool3, [-1, 1])
            out = tf.real((out))
            return out

    def build_FCN3_1d(self, x):
        act = tf_.soft_plus2
        with tf.variable_scope("network", reuse=None):
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
            # pool4 = tf.exp(pool4)
            out = tf.reshape(pool4, [-1, 1])

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
            return out

    def build_ResNet(self, x):
        with tf.variable_scope("network", reuse=None):
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

        return out

    def build_RBM_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            # inputShape = x.get_shape().as_list()
            x = x[:, :, 0]
            fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.soft_plus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias_re = tf_.fc_layer(x, self.L, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, self.L, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out

    def build_RBM_cosh_1d(self, x):
        with tf.variable_scope("network", reuse=None):
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

        return out


    def build_sRBM_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_1d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.soft_plus2(tf.complex(conv1_re, conv1_im))
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
            return out

    def build_NN_complex(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1_complex',
                                       dtype=tf.complex64)
            fc1_complex = tf_.soft_plus(fc1_complex)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, 1, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)

            out = tf.exp(fc2_complex)
            out = tf.real(out)

        return out

    def build_NN3_complex(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1_complex = tf_.fc_layer(tf.complex(x, 0.), self.L, self.L * self.alpha, 'fc1_complex',
                                       dtype=tf.complex64)
            fc1_complex = tf_.soft_plus(fc1_complex)
            # fc1_complex = fc1_complex + tf.complex(x, 0.)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, self.L * self.alpha, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)
            fc2_complex = tf_.soft_plus(fc2_complex)
            # fc2_complex = fc2_complex + fc1_complex

            fc3_complex = tf_.fc_layer(fc2_complex, self.L * self.alpha, self.L * self.alpha, 'fc3_complex',
                                       dtype=tf.complex64, biases=True)
            fc3_complex = tf_.soft_plus(fc3_complex)
            # fc3_complex = fc3_complex  + fc2_complex

            # fc4_complex = tf.reduce_sum(fc3_complex, axis=1, keep_dims=True)
            # out = tf.multiply(tf.exp(tf.real(fc4_complex) / self.L), tf.cos(tf.imag(fc4_complex)))
            fc4_complex = tf_.fc_layer(fc3_complex, self.L * self.alpha, 1, 'fc4_complex',
                                       dtype=tf.complex64, biases=True)
            out = tf.exp(fc4_complex)
            out = tf.real(out)

        return out

    def build_RBM_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            inputShape = x.get_shape().as_list()
            b_size, Lx, Ly, _ = inputShape
            LxLy = Lx * Ly
            x = tf.reshape(x[:, :, :, 0], [-1, LxLy])
            fc1_re = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, LxLy, LxLy * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.soft_plus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias_re = tf_.fc_layer(x, LxLy, 1, 'v_bias_re')
            v_bias_im = tf_.fc_layer(x, LxLy, 1, 'v_bias_im')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias_re, v_bias_im))
            out = tf.real(tf.exp(log_prob))

        return out

    def build_RBM_cosh_2d(self, x):
        with tf.variable_scope("network", reuse=None):
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

        return out

    def build_sRBM_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=2, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=2, biases=True, bias_scale=300., FFT=False)

            conv1 = tf_.soft_plus2(tf.complex(conv1_re, conv1_im))
            pool4 = tf.reduce_sum(conv1, [1, 2, 3], keep_dims=False)
            pool4 = tf.exp(pool4)

            # conv1 = tf.cosh(tf.complex(conv1_re, conv1_im))
            # pool4 = tf.reduce_prod(conv1, [1, 2], keep_dims=False)

            conv_bias_re = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_re',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias_im = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_im',
                                                stride_size=2, bias_scale=100., FFT=False)
            conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
                                      [1, 2, 3], keep_dims=False)
            out = tf.reshape(tf.multiply(pool4, tf.exp(conv_bias)), [-1, 1])
            out = tf.real((out))
            return out

    def build_FCN2_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, :, 0:1]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer2d(x, filter_size, in_channels, out_channels, name)
            conv1_re = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_re',
                                            stride_size=1, biases=True, bias_scale=100., FFT=False)
            conv1_im = tf_.circular_conv_2d(x, inputShape[1], inputShape[-1], self.alpha, 'conv1_im',
                                            stride_size=1, biases=True, bias_scale=300., FFT=False)
            conv1 = tf_.soft_plus2(tf.complex(conv1_re, conv1_im))

            conv2 = tf_.circular_conv_2d_complex(conv1, inputShape[1], self.alpha, self.alpha*2,
                                                 'conv2_complex', stride_size=2, biases=True,
                                                 bias_scale=100.)
            conv2 = tf_.soft_plus2(conv2)

            pool3 = tf.reduce_sum(conv2, [1, 2, 3], keep_dims=False)
            pool3 = tf.exp(pool3)

            ## Conv Bias
            # conv_bias_re = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_re',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias_im = tf_.circular_conv_2d(x, 2, inputShape[-1], 1, 'conv_bias_im',
            #                                     stride_size=2, bias_scale=100.)
            # conv_bias = tf.reduce_sum(tf.complex(conv_bias_re, conv_bias_im),
            #                           [1, 2], keep_dims=False)
            # out = tf.reshape(tf.multiply(pool3, tf.exp(conv_bias)), [-1, 1])
            out = tf.reshape(pool3, [-1, 1])
            out = tf.real((out))
            return out


    def build_Jastrow_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, :, :]
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # def jastrow_2d_amp(config_array, Lx, Ly, local_d, name, sym=False):
            out = tf_.jastrow_2d_amp(x, inputShape[1], inputShape[2], inputShape[-1], 'jastrow')
            out = tf.real((out))
            return out


    def build_network_1d(self, which_net, x):
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

    def build_network_2d(self, which_net, x):
        if which_net == "NN":
            return self.build_NN_2d(x)
        elif which_net == "NN3":
            return self.build_NN3_2d(x)
        elif which_net == "RBM":
            return self.build_RBM_2d(x)
        elif which_net == "RBM_cosh":
            return self.build_RBM_cosh_2d(x)
        elif which_net == "sRBM":
            return self.build_sRBM_2d(x)
        elif which_net == "FCN2":
            return self.build_FCN2_2d(x)
        elif which_net == "Jastrow":
            return self.build_Jastrow_2d(x)
        else:
            raise NotImplementedError
