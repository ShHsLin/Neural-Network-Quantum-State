import numpy as np
import tensorflow as tf
import tf_wrapper as tf_


class tf_NN3_complex:
    def __init__(self, inputShape, optimizer, learning_rate=0.1125,
                 momentum=0.90, alpha=1):
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
            'wd1': tf.Variable(tf.random_normal([(self.L), (self.L * alpha)],
                                                stddev=np.sqrt(2./(self.L*(1+alpha))))),
            'wd2': tf.Variable(tf.random_normal([(self.L*alpha), (self.L * alpha)],
                                                stddev=np.sqrt(2./(self.L*alpha*2)))),
            'wd3_re': tf.Variable(tf.random_normal([(self.L * alpha), (self.L * alpha)],
                                                   stddev=np.sqrt(2./(self.L * 2 * alpha)/100.))),
            'wd3_im': tf.Variable(tf.random_normal([(self.L * alpha), (self.L * alpha)],
                                                   stddev=np.sqrt(2./(self.L * 2 * alpha)))),
            'out_re': tf.Variable(tf.random_normal([self.L*alpha, n_classes],
                                                   stddev=np.sqrt(2./(self.L*alpha)))),
            'out_im': tf.Variable(tf.random_normal([self.L*alpha, n_classes],
                                                   stddev=np.sqrt(2./(self.L*alpha))))
        }

        self.biases = {
            'bd1': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'bd2': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'bd3_re': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'bd3_im': tf.Variable(np.zeros(self.L*alpha, dtype=np.float32)),
            'out': tf.Variable((np.zeros(1, dtype=np.float32)))
        }

        # Construct model : Tensorflow Graph is built here !
        self.pred = self.build(self.x, self.weights, self.biases, inputShape)

        self.model_var_list = tf.global_variables()

        # Define optimizer
        self.optimizer = tf_.select_optimizer(optimizer, self.learning_rate,
                                              self.momentum)

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
    def build(self, x, weights, biases, inputShape):
        x = tf.reshape(x, shape=[-1, inputShape[0], inputShape[1], 1])
        x = x[:, :, 0, :]
        # Fully connected layer
        fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)

        fc3 = tf.add(tf.matmul(tf.complex(fc2, 0.),
                               tf.complex(weights['wd3_re'],
                                          weights['wd3_im'])),
                     tf.complex(biases['bd3_re'], biases['bd3_im']))
        fc3 = tf.exp(fc3)

        out = tf.add(tf.matmul(fc3, tf.complex(weights['out_re'],
                                               weights['out_im'])),
                     tf.complex(biases['out'], 0.0))
        out = tf.real(out)
        #    out = tf.nn.sigmoid(out)
        print("Building the model with shape:")
        print("Input Layer X:", x.get_shape())
        print("FC1:", fc1.get_shape())
        print("out:", out.get_shape())
        return out
