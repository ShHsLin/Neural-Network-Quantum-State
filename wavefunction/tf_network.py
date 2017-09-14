import numpy as np
import tensorflow as tf
import tf_wrapper as tf_


class tf_network:
    def __init__(self, which_net, inputShape, optimizer, learning_rate=0.1125,
                 momentum=0.90, alpha=2):
        # Parameters
        self.learning_rate = tf.Variable(learning_rate)
        self.momentum = tf.Variable(momentum)
        # dropout = 0.75  # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, inputShape[0], inputShape[1]])
        self.keep_prob = tf.placeholder(tf.float32)

        self.L = int(inputShape[0])
        self.alpha = alpha

        # Variables Creation
        self.pred = self.build_network(which_net, self.x)
        self.model_var_list = tf.global_variables()
        self.para_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
        print("create variable")
        for i in self.para_list:
            print i.name
        # Define optimizer
        self.optimizer = tf_.select_optimizer(optimizer, self.learning_rate,
                                              self.momentum)

        # Define Gradient, loss = log(wave function)
        self.grads = tf.gradients(tf.log(self.pred), self.para_list)  # grad(cost, variable_list)
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
        # print(self.sess.run(self.para_list[2])[:10])
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    def build_NN_1d(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = tf.nn.tanh(fc1)
            out_re = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc1, self.L * self.alpha, 1, 'out_im')
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

    def build_CNN_1d(self, x):
        inputShape = x.get_shape().as_list()
        # x_shape = [num_data, num_site, num_spin(channels)]
        x = tf.reshape(x, [-1, inputShape[1], 1, inputShape[2]])
        return self.build_CNN_2d(x)

    def build_CNN_2d(self, x):
        with tf.variable_scope("network", reuse=None):
            inputShape = x.get_shape().as_list()
            # x_shape = [num_data, Lx, Ly, num_spin(channels)]
            # conv_layer1d(x, filter_size, in_channels, out_channels, name)
            conv1 = tf_.conv_layer2d(x, 3, inputShape[-1], self.alpha, 'conv1')
            conv1 = tf_.leaky_relu(conv1)

            conv2 = tf_.conv_layer2d(conv1, 3, self.alpha, self.alpha * 2, 'conv2')
            conv2 = tf_.leaky_relu(conv2)

            pool2 = tf_.avg_pool2d(conv2, 'pool2', 2)

            conv3 = tf_.conv_layer2d(pool2, 3, self.alpha * 2, self.alpha * 2, 'conv3')
            conv3 = tf_.leaky_relu(conv3)

            conv4 = tf_.conv_layer2d(conv3, 3, self.alpha * 2, self.alpha * 2, 'conv4')
            conv4 = tf_.leaky_relu(conv4)

            pool4 = tf_.avg_pool2d(conv4, 'pool4', 2)
            # pool2 = tf_.avg_pool1d(

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc_dim = np.prod(pool4.get_shape().as_list()[1:])
            pool4 = tf.reshape(pool4, [-1, fc_dim])
            out_re = tf_.fc_layer(pool4, fc_dim, 1, 'out_re')
            out_im = tf_.fc_layer(pool4, fc_dim, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
            return out

    def build_ResNet(self, x):
        with tf.variable_scope("network", reuse=None):
            x = x[:, :, 0]
            fc1 = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1')
            fc1 = fc1 + x
            fc1 = tf.nn.softplus(fc1)

            fc2 = tf_.fc_layer(fc1, self.L * self.alpha, self.L * self.alpha, 'fc2')
            fc2 = fc2 + fc1
            fc2 = tf.nn.softplus(fc2)

            fc3 = tf_.fc_layer(fc2, self.L * self.alpha, self.L * self.alpha, 'fc3')
            fc3 = fc3 + fc2
            fc3 = tf.nn.softplus(fc3)

            out_re = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_re')
            out_im = tf_.fc_layer(fc3, self.L * self.alpha, 1, 'out_im')
            out = tf.multiply(tf.exp(out_re), tf.cos(out_im))

        return out

    def build_RBM(self, x):
        with tf.variable_scope("network", reuse=None):
            # inputShape = x.get_shape().as_list()
            x = x[:, :, 0]
            fc1_re = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_re')
            fc1_im = tf_.fc_layer(x, self.L, self.L * self.alpha, 'fc1_im')
            fc1 = tf.complex(fc1_re, fc1_im)
            fc2 = tf_.soft_plus2(fc1)
            # fc2 = tf_.complex_relu(fc1)

            v_bias = tf_.fc_layer(x, self.L, 1, 'v_bias')
            log_prob = tf.reduce_sum(fc2, axis=1, keep_dims=True)
            log_prob = tf.add(log_prob, tf.complex(v_bias, 0.0))
            out = tf.real(tf.exp(log_prob))

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
            fc1_complex = fc1_complex + tf.complex(x, 0.)

            fc2_complex = tf_.fc_layer(fc1_complex, self.L * self.alpha, self.L * self.alpha, 'fc2_complex',
                                       dtype=tf.complex64, biases=True)
            fc2_complex = tf_.soft_plus(fc2_complex)
            fc2_complex = fc2_complex + fc1_complex

            fc3_complex = tf_.fc_layer(fc2_complex, self.L * self.alpha, self.L * self.alpha, 'fc3_complex',
                                       dtype=tf.complex64, biases=True)
            fc3_complex = tf_.soft_plus(fc3_complex)
            fc3_complex = fc3_complex  # + fc2_complex

            # fc4_complex = tf.reduce_sum(fc3_complex, axis=1, keep_dims=True)
            # out = tf.multiply(tf.exp(tf.real(fc4_complex) / self.L), tf.cos(tf.imag(fc4_complex)))
            fc4_complex = tf_.fc_layer(fc3_complex, self.L * self.alpha, 1, 'fc4_complex',
                                       dtype=tf.complex64, biases=True)
            out = tf.exp(fc4_complex)
            out = tf.real(out)

        return out

    def build_network(self, which_net, x):
        if which_net == "NN":
            return self.build_NN_1d(x)
        elif which_net == "NN3":
            return self.build_NN3_1d(x)
        elif which_net == "CNN":
            return self.build_CNN_1d(x)
        elif which_net == "FCN":
            return self.build_FCN(x)
        elif which_net == "NN_complex":
            return self.build_NN_complex(x)
        elif which_net == "NN3_complex":
            return self.build_NN3_complex(x)
        elif which_net == "RBM":
            return self.build_RBM(x)
        elif which_net == "ResNet":
            return self.build_ResNet(x)
        else:
            raise NotImplementedError
