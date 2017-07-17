import numpy as np
import tensorflow as tf
from tf_wrapper import *


class tf_NN:
    def __init__(self, inputShape, optimizer, learning_rate=0.1125,
                 momentum=0.90, alpha=2):
        # Parameters
        self.learning_rate = tf.Variable(learning_rate)
        self.momentum = tf.Variable(momentum)
        # Network Parameters
        n_input = int(inputShape[0]*inputShape[1])
        n_classes = 1
        # dropout = 0.75  # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        self.L = int(inputShape[0])

        # Variables Creation
        # Store layers weight & bias
        self.weights = {
            'wd1': tf.Variable(tf.random_normal([(self.L), (self.L * alpha)],
                                                stddev=np.sqrt(2./(self.L*(1+alpha))))),
            'out_re': tf.Variable(tf.random_normal([self.L*alpha, n_classes],
                                                   stddev=np.sqrt(2./(self.L*alpha)))),
            'out_im': tf.Variable(tf.random_normal([self.L*alpha, n_classes],
                                                   stddev=np.sqrt(2./(self.L*alpha))))
        }

        self.biases = {
            'bd1': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'out_re': tf.Variable(np.zeros(1, dtype=np.float32)),
            'out_im': tf.Variable(np.zeros(1, dtype=np.float32))
        }

        # Construct model : Tensorflow Graph is built here !
        self.pred = self.build(self.x, self.weights, self.biases,
                               self.keep_prob, inputShape)

        self.model_var_list = tf.global_variables()

        # Define optimizer
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif optimizer == 'Mom':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=self.momentum)
        elif optimizer == 'RMSprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise

        # Define Gradient, loss = log(wave function)
        self.para_list = self.weights.values() + self.biases.values()
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
        X0 = X0.reshape(X0.size/(self.L*2), self.L*2)
        return self.sess.run(self.pred, feed_dict={self.x: X0, self.keep_prob: 1.})

    def backProp(self, X0):
        X0 = X0.reshape(X0.size/(self.L*2), self.L*2)
        return self.sess.run(self.grads, feed_dict={self.x: X0, self.keep_prob: 1.})

    def getNumPara(self):
        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        # print(self.sess.run(self.para_list[2])[:10])
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    # Create model
    def build(self, x, weights, biases, dropout, inputShape):
        x = tf.reshape(x, shape=[-1, inputShape[0], inputShape[1], 1])
        x = x[:, :, 0, :]
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.tanh(fc1)
        # fc1 = tf.cos(fc1)

        out_re = tf.add(tf.matmul(fc1, weights['out_re']), biases['out_re'])
        out_im = tf.add(tf.matmul(fc1, weights['out_im']), biases['out_im'])
        out = tf.multiply(tf.exp(out_re), tf.cos(out_im))
        #    out = tf.nn.sigmoid(out)
        print("Building the model with shape:")
        print("Input Layer X:", x.get_shape())
        print("FC1:", fc1.get_shape())
        print("out:", out.get_shape())
        return out
