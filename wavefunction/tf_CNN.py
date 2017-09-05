import numpy as np
import tensorflow as tf
import tf_wrapper as tf_


class tf_CNN:
    def __init__(self, inputShape, optimizer, learning_rate=0.1125,
                 momentum=0.95):
        # Parameters
        self.learning_rate = tf.Variable(learning_rate)
        self.momentum = tf.Variable(momentum)
        # Network Parameters
        # dropout = 0.75  # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, inputShape[0], inputShape[1]])
        self.keep_prob = tf.placeholder(tf.float32)

        self.L = int(inputShape[0])

        # Variables Creation
        # Store layers weight & bias
        chan1 = 20
        self.weights = {
            'wc1': tf.Variable(tf.random_normal([4, 2, 1, chan1], stddev=0.2)),
            'wd1': tf.Variable(tf.random_normal([(self.L)*1*chan1, 5],
                                                stddev=np.sqrt(2./((self.L)*1*chan1+5)))),
            'out': tf.Variable(tf.random_normal([5, 1], stddev=0.1))
        }

        self.biases = {
            'bc1': tf.Variable(np.zeros(chan1, dtype=np.float32)),
            'bd1': tf.Variable(np.zeros(5, dtype=np.float32)),
            'out': tf.Variable(np.zeros(1, dtype=np.float32))
        }

        # Construct model : Tensorflow Graph is built here !
        self.pred = self.conv_net(self.x, self.weights, self.biases,
                                  self.keep_prob, inputShape)

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
        return self.sess.run(self.pred, feed_dict={self.x: X0, self.keep_prob: 1.})

    def backProp(self, X0):
        return self.sess.run(self.grads, feed_dict={self.x: X0, self.keep_prob: 1.})

    def getNumPara(self):
        return sum([np.prod(w.get_shape().as_list()) for w in self.para_list])

    def applyGrad(self, grad_list):
        # print(self.sess.run(self.para_list[2])[:10])
        self.sess.run(self.train_op, feed_dict={i: d for i, d in
                                                zip(self.newgrads, grad_list)})

    # Create model
    def conv_net(self, x, weights, biases, dropout, inputShape):
        x = tf.reshape(x, shape=[-1, inputShape[0], inputShape[1], 1])

        conv1 = tf_.conv2d(x, weights['wc1'], biases['bc1'], padding='SAME')
        conv2 = tf_.soft_plus2(conv1)
        # conv2 = leaky_relu(conv1)
        # conv2 = tf.cos(conv1)
        # conv2_2 = tf.cos(conv1)
        # conv2 = tf.multiply(conv2, conv2_2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        # fc1 = tf.nn.tanh(fc1)
        fc1 = tf.exp(fc1)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        #    out = tf.nn.sigmoid(out)
        print("Building the model with shape:")
        print("Input Layer X:", x.get_shape())
        print("Conv1:", conv1.get_shape())
        print("FC1:", fc1.get_shape())
        print("out:", out.get_shape())
        return out
