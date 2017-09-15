import tensorflow as tf
from functools import reduce
import numpy as np


def select_optimizer(optimizer, learning_rate, momentum=0):
    if optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'Mom':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=momentum)
    elif optimizer == 'RMSprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         epsilon=0.1)
    elif optimizer == 'GD':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        raise


def leaky_relu(x):
    return tf.maximum(0.01*x, x)
    # return tf.nn.sigmoid(x)
    # return tf.nn.tanh(x)


def soft_plus(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x)))


def soft_plus2(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x))/2.)


def complex_relu(x):
    re = tf.real(x)
    im = tf.imag(x)
    mask = tf.cast(tf.greater(re, tf.zeros_like(re)), tf.float32)
    re = re * mask
    im = im * mask  # if re>0; im*1; else: im*0
    return tf.complex(re, im)


def complex_relu2(x):
    return tf.complex(tf.nn.relu(tf.real(x)), tf.imag(x))


def max_pool1d(x, name, kernel_size=2, stride_size=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, 1, 1],
                          strides=[1, stride_size, 1, 1],
                          padding=padding, name=name)


def max_pool2d(x, name, kernel_size=2, stride_size=2, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding, name=name)


def avg_pool1d(x, name, kernel_size, stride_size=2, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, 1, 1],
                          strides=[1, stride_size, 1, 1],
                          padding=padding, name=name)


def avg_pool2d(x, name, kernel_size, stride_size=2, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding, name=name)


def batch_norm(bottom, phase, scope='bn'):
    return tf.contrib.layers.batch_norm(bottom, center=True, scale=True,
                                        is_training=phase, scope=scope,
                                        decay=0.995)


def conv_layer1d(bottom, filter_size, in_channels,
                 out_channels, name, stride_size=1, biases=False):
    with tf.variable_scope(name, reuse=None):
        filt, conv_biases = get_conv_var1d(filter_size, in_channels,
                                           out_channels, biases=biases)
        conv = tf.nn.conv1d(bottom, filt, stride_size, padding='SAME')
        if not biases:
            return conv
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def conv_layer2d(bottom, filter_size, in_channels,
                 out_channels, name, stride_size=1, biases=False):
    with tf.variable_scope(name, reuse=None):
        filt, conv_biases = get_conv_var2d(filter_size, in_channels,
                                           out_channels, biases=biases)
        conv = tf.nn.conv2d(bottom, filt, [1, stride_size, stride_size, 1], padding='SAME')
        if not biases:
            return conv
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def fc_layer(bottom, in_size, out_size, name, biases=True, dtype=tf.float32):
    if dtype not in [tf.complex64, tf.complex128]:
        with tf.variable_scope(name, reuse=None):
            weights, biases = get_fc_var(in_size, out_size, biases=biases, dtype=dtype)
            x = tf.reshape(bottom, [-1, in_size])
            if biases:
                fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            else:
                fc = tf.matmul(x, weights)

    else:
        part_dtype = {tf.complex64: tf.float32, tf.complex128: tf.float64}
        with tf.variable_scope(name, reuse=None):
            real_weights, real_biases = get_fc_var(in_size, out_size, name="real_",
                                                   biases=biases, dtype=part_dtype[dtype])
            imag_weights, imag_biases = get_fc_var(in_size, out_size, name="imag_",
                                                   biases=biases, dtype=part_dtype[dtype])
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, tf.complex(real_weights, imag_weights))
            # real_fc = tf.matmul(tf.real(x), real_weights) - tf.matmul(tf.imag(x), imag_weights)
            # imag_fc = tf.matmul(tf.real(x), imag_weights) + tf.matmul(tf.imag(x), real_weights)
            if biases:
                fc = tf.nn.bias_add(fc, tf.complex(real_biases, imag_biases))
                # fc = tf.nn.bias_add(tf.complex(real_fc, imag_fc),
                #                     tf.complex(real_biases, imag_biases))
            else:
                pass
                # fc = tf.complex(real_fc, imag_fc)

    return fc


def get_conv_var1d(filter_size, in_channels, out_channels, name="",
                   biases=False, dtype=tf.float32):
    if dtype == tf.complex64:
        raise NotImplementedError
        # tensorflow optimizer does not support complex type
    else:
        pass

    initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0, 0.001)
    filters = get_var(initial_value, name + "weights", dtype=dtype)

    if not biases:
        return filters, None
    else:
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(initial_value, name + "biases", dtype=dtype)
        return filters, biases


def get_conv_var2d(filter_size, in_channels, out_channels, name="",
                   biases=False, dtype=tf.float32):
    if dtype == tf.complex64:
        raise NotImplementedError
        # tensorflow optimizer does not support complex type
    else:
        pass

    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0,
                                        0.01)
    # initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0,
    #                                     np.sqrt(filter_size*filter_size*(in_channels+out_channels)))
    filters = get_var(initial_value, name + "weights", dtype=dtype)

    if not biases:
        return filters, None
    else:
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = get_var(initial_value, name + "biases", dtype=dtype)
        return filters, biases


def get_fc_var(in_size, out_size, name="", biases=True, dtype=tf.float32):
    # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
    if dtype in [tf.complex64, tf.complex128]:
        raise NotImplementedError
        # tensorflow optimizer does not support complex type
    else:
        pass

    # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1)
    initial_value = tf.random_normal([in_size, out_size], stddev=np.sqrt(2./(in_size+out_size)))
    weights = get_var(initial_value, name + "weights", dtype=dtype)

    if biases:
        # initial_value = tf.truncated_normal([out_size], .0, .001, dtype=dtype)
        initial_value = tf.zeros(out_size, dtype=dtype)
        biases = get_var(initial_value, name + "biases", dtype=dtype)
        return weights, biases
    else:
        return weights, None


def get_var(initial_value, var_name, dtype):
    # if self.is_training:
    var = tf.get_variable(var_name, initializer=initial_value, trainable=True, dtype=dtype)
    # self.var_dict[var.name] = var
    # print var_name, var.get_shape().as_list()
    assert var.get_shape() == initial_value.get_shape()
    return var


def save_npy(self, sess, npy_path="./vgg19-save.npy"):
    assert isinstance(sess, tf.Session)

    data_dict = {}

    for (name, idx), var in list(self.var_dict.items()):
        var_out = sess.run(var)
        if name not in data_dict:
            data_dict[name] = {}

        data_dict[name][idx] = var_out

    np.save(npy_path, data_dict)
    print(("file saved", npy_path))
    return npy_path


def get_var_count(self):
    count = 0
    for v in list(self.model_var_list):
        count += reduce(lambda x, y: x * y, v.get_shape().as_list())
    print("Total parameter, including auxiliary variables: %d\n" % count)

    count = 0
    for v in list(self.var_dict.values()):
        count += reduce(lambda x, y: x * y, v.get_shape().as_list())

    return count


def bottleneck_residual(self, x, in_channel, out_channel, name,
                        stride_size=2):
    with tf.variable_scope(name, reuse=None):
        # Identity shortcut
        if in_channel == out_channel:
            shortcut = x
            x = self.conv_layer(x, 1, in_channel, out_channel/4, "conv1")
            # conv projection shortcut
        else:
            shortcut = x
            shortcut = self.conv_layer(shortcut, 1, in_channel,
                                       out_channel, "shortcut",
                                       stride_size=stride_size)
            shortcut = self.batch_norm(shortcut, phase=self.bn_is_training,
                                       scope='shortcut/bn')
            x = self.conv_layer(x, 1, in_channel, out_channel/4, "conv1",
                                stride_size=stride_size)

        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
        x = tf.nn.relu(x)
        x = self.conv_layer(x, 3, out_channel/4, out_channel/4, "conv2")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn2')
        x = tf.nn.relu(x)
        x = self.conv_layer(x, 1, out_channel/4, out_channel, "conv3")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn3')
        x += shortcut
        x = tf.nn.relu(x)

    return x
