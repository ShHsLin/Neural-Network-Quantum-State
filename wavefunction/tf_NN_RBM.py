import numpy as np
import tensorflow as tf
from tf_wrapper import *


class tf_NN_RBM:
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
        self.y = tf.placeholder(tf.complex64, [None, n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        self.L = int(inputShape[0])

        # Variables Creation
        # Store layers weight & bias
        self.weights = {
            'wd1_re': tf.Variable(tf.random_normal([(self.L), (self.L * alpha)],
                                                   stddev=1e-4*np.sqrt(2./(self.L*(1+alpha))))),
            'wd1_im': tf.Variable(tf.random_normal([(self.L), (self.L * alpha)],
                                                   stddev=np.sqrt(2./(self.L*(1+alpha))))),
            # 'wd2': tf.Variable(tf.random_normal([self.L, 1], stddev=0.01))
        }

        self.biases = {
            'bd1_re': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'bd1_im': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
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
        fc1 = tf.reshape(x, [-1, weights['wd1_re'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(tf.complex(fc1, 0.),
                               tf.complex(weights['wd1_re'],
                                          weights['wd1_im'])),
                     tf.complex(biases['bd1_re'], biases['bd1_im']))
        # fc1 = tf.nn.tanh(fc1)
        fc1 = tf.exp(fc1)
        fc2 = tf.add(tf.ones_like(fc1), fc1)
        fc2 = tf.divide(fc2,2)
        # rad = tf.abs(fc2)
        # angle = tf.acos(tf.divide(tf.real(fc2), rad))
        # out = tf.multiply(tf.reduce_prod(rad,axis=1),
        #                   tf.cos(tf.reduce_sum(angle,axis=1)))
        fc2 = tf.log(fc2)

        # v_bias =  tf.reshape(x, [-1, weights['wd2'].get_shape().as_list()[0]])
        # v_bias = tf.matmul(v_bias, weights['wd2'])
        # fc2 = tf.add(fc2, tf.complex(v_bias, 0.0))

        out = tf.real(tf.exp(tf.reduce_sum(fc2, axis=1, keep_dims=True)))
        # fc2_re = tf.real(fc2)
        # fc2_im = tf.imag(fc2)
        # out = tf.multiply(tf.exp(tf.reduce_sum(fc2_re)),
        #                   tf.exp(tf.complex(0.0,tf.reduce_sum(fc2_imag)))


#        out = tf.add(tf.matmul(fc1, tf.complex(weights['out_re'],
#                                               weights['out_im'])),
#                     tf.complex(biases['out'], 0.0))
#        out = tf.real(out)
        #    out = tf.nn.sigmoid(out)
        print("Building the model with shape:")
        print("Input Layer X:", x.get_shape())
        print("FC1:", fc1.get_shape())
        print("out:", out.get_shape())
        return out