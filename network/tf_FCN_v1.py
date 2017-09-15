import tensorflow as tf
import tf_wrapper as tf_
import numpy as np


class tf_FCN:
    def __init__(self, inputShape, optimizer, learning_rate=0.1125,
                 momentum=0.95, alpha=2):
        # Parameters
        self.learning_rate = tf.Variable(learning_rate)
        self.momentum = tf.Variable(momentum)
        # dropout = 0.75  # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, inputShape[0], inputShape[1]])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        self.L = int(inputShape[0])

        chan1 = 2 * alpha
        chan2 = 2 * alpha
        chan3 = 4 * alpha
        chan4 = 4 * alpha
        chan5 = 6 * alpha
        chan6 = 6 * alpha
        chan7 = 8 * alpha
        chan8 = 8 * alpha

        self.weights = {
            'wc1': tf.Variable(tf.random_normal([4, 2, 1, chan1],
                                                stddev=np.sqrt(2./8))),
            'wc2': tf.Variable(tf.random_normal([2, 1, chan1, chan2],
                                                stddev=np.sqrt(2./8))),
            'wc3': tf.Variable(tf.random_normal([2, 1, chan2, chan3],
                                                stddev=np.sqrt(2./4))),
            'wc4': tf.Variable(tf.random_normal([2, 1, chan3, chan4],
                                                stddev=np.sqrt(2./8))),
            'wc5': tf.Variable(tf.random_normal([2, 1, chan4, chan5],
                                                stddev=np.sqrt(2./8))),
            'wc6': tf.Variable(tf.random_normal([2, 1, chan5, chan6],
                                                stddev=np.sqrt(2./12))),
            'wc7': tf.Variable(tf.random_normal([2, 1, chan6, chan7],
                                                stddev=np.sqrt(2./12))),
            'wc8': tf.Variable(tf.random_normal([2, 1, chan7, chan8],
                                                stddev=np.sqrt(2./16))),
            # fully connected, Lpp*Lpp*8 inputs, 5 outputs
            # 'wd1': tf.Variable(tf.random_normal([self.L*chan5/4, 4], stddev = 0.1)),
            # 'wd1': tf.Variable(tf.random_uniform([(self.L-3)*1*6, 5])/(self.L-3)),
            # 5 inputs, 1 outputs (class prediction)
            # 'out': tf.Variable(tf.random_normal([4, n_classes], stddev = 0.4))
            'out': tf.Variable(tf.random_normal([self.L / 16 * chan8, 1]))
            # 'out': tf.Variable(tf.random_normal([L_pool*1*chan3,
            #                                     n_classes])/10 )
        }

        self.biases = {
            'bc1': tf.Variable(np.zeros([chan1], dtype=np.float32)),
            'bc2': tf.Variable(np.zeros([chan2], dtype=np.float32)),
            'bc3': tf.Variable(np.zeros([chan3], dtype=np.float32)),
            'bc4': tf.Variable(np.zeros([chan4], dtype=np.float32)),
            'bc5': tf.Variable(np.zeros([chan5], dtype=np.float32)),
            'bc6': tf.Variable(np.zeros([chan6], dtype=np.float32)),
            'bc7': tf.Variable(np.zeros([chan7], dtype=np.float32)),
            'bc8': tf.Variable(np.zeros([chan8], dtype=np.float32))
            # 'bd1': tf.Variable(np.zeros(5,dtype=np.float32)),
            # 'bd1': tf.Variable(tf.random_normal([4],stddev=0.1)),
            # 'out': tf.Variable(tf.random_normal([n_classes],stddev=0.1))
        }

        # Construct model
        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob, inputShape)

        self.model_var_list = tf.global_variables()

        # Define loss and optimizer
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif optimizer == 'Mom':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=self.momentum)
        elif optimizer == 'RMSprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=self.momentum)

        # Define a list of names of parameters
        self.para_list = self.weights.values()  # +self.biases.values()

        # Define Gradient
        self.grads = tf.gradients(tf.log(self.pred), self.para_list)  # pred --> cost
        # Do some operation on grads
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
        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    def conv_net(self, x, weights, biases, dropout, inputShape):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, inputShape[0], inputShape[1], 1])

        conv1 = tf_.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = tf_.leaky_relu(conv1)
        conv2 = tf_.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = tf_.leaky_relu(conv2)
        conv2 = tf_.avgpool1d(conv2, k=2)
        #    conv2 = maxpool2d(conv2, k=2)

        conv3 = tf_.conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = tf_.leaky_relu(conv3)
        conv4 = tf_.conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = tf_.leaky_relu(conv4)
        conv4 = tf_.avgpool1d(conv4, k=2)
        #    conv4 = maxpool1d(conv4, k=2)

        conv5 = tf_.conv2d(conv4, weights['wc5'], biases['bc5'])
        conv5 = tf_.leaky_relu(conv5)
        conv6 = tf_.conv2d(conv5, weights['wc6'], biases['bc6'])
        conv6 = tf_.leaky_relu(conv6)
        conv6 = tf_.avgpool1d(conv6, k=2)
        #    conv6 = maxpool1d(conv6, k=2)

        conv7 = tf_.conv2d(conv6, weights['wc7'], biases['bc7'])
        conv7 = tf_.leaky_relu(conv7)
        conv8 = tf_.conv2d(conv7, weights['wc8'], biases['bc8'])
        conv8 = tf_.leaky_relu(conv8)
        conv8 = tf_.avgpool1d(conv8, k=2)
        #    conv8 = maxpool1d(conv8, k=2)
        conv8 = tf.exp(conv8)

        # Fully connected layer
        # fc1 = tf.reshape(conv8, [1, 32])
        fc1 = tf.reshape(conv8, [-1, weights['out'].get_shape().as_list()[0]])
        # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        # fc1 = tf.nn.sigmoid(fc1)
        # Apply Dropout
        # fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        # out = tf.matmul(fc1, np.ones((32,1),dtype=np.float32))
        out = tf.matmul(fc1, weights['out'])
        #    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        #    out = tf.nn.sigmoid(out)
        #    out = tf.nn.tanh(out)
        return out
