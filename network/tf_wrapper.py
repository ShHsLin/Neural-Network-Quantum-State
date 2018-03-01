import tensorflow as tf
from functools import reduce
import numpy as np


def select_optimizer(optimizer, learning_rate, momentum=0):
    if optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      epsilon=1e-3) #previous 1e-8
    elif optimizer == 'Mom':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=momentum)
    elif optimizer == 'RMSprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         epsilon=1e-6) #previous 1e-1
    elif optimizer == 'GD':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                          epsilon=1e-6)
    else:
        raise

def select_activation(activation):
    if activation == 'softplus2':
        return softplus2
    elif activation == 'softplus':
        return softplus
    elif activation == 'c_relu':
        return c_relu
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'c_elu':
        return c_elu
    elif activation == 'elu':
        return elu
    elif activation == 'tanh':
        return tf.tanh
    elif activation == 'selu':
        return tf.nn.selu


def leaky_relu(x):
    return tf.maximum(0.01*x, x)

def softplus(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x)))


def softplus2(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x))/2.)


def complex_relu(x):
    re = tf.real(x)
    # im = tf.imag(x)
    return tf.where(tf.greater(re, tf.zeros_like(re)), x, tf.complex(tf.nn.elu(re),
                                                                     tf.zeros_like(re)))
    # mask = tf.cast(tf.greater(re, tf.zeros_like(re)), tf.float32)
    # re = re * mask
    # im = im * mask  # if re>0; im*1; else: im*0
    # return tf.complex(re, im)

def c_relu(x):
    return tf.complex(tf.nn.relu(tf.real(x)), tf.nn.relu(tf.imag(x)))

def c_elu(x):
    return tf.complex(tf.nn.elu(tf.real(x)), tf.nn.elu(tf.imag(x)))


# def complex_relu(x):
#     return tf.complex(tf.nn.relu(tf.real(x)), tf.imag(x))


def complex_relu_m1(x):
    return tf.complex(tf.nn.relu(tf.real(x)) - 1.0, tf.imag(x))


def complex_relu_neg(x):
    return tf.complex(tf.maximum(-0.7, tf.real(x)), tf.imag(x))


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


def circular_conv_1d(bottom, filter_size, in_channels, out_channels,
                     name, stride_size=1, biases=False, bias_scale=1.,
                     FFT=False):
    '''
    FFT can be used instead of circular convolution. Although, the pad and conv
    approach is relatively slow comparing to ordinary conv operation, it is still
    much faster than FFT approach generally, since the dimension of the filter is small.
    '''
    with tf.variable_scope(name, reuse=None):
        filt, conv_biases = get_conv_var1d(filter_size, in_channels,
                                           out_channels, biases=biases, bias_scale=bias_scale)
        if not FFT:
            # bottom shape [None, Lx, channels]
            # pad_size = filter_size - 1
            bottom_pad = tf.concat([bottom, bottom[:, :filter_size-1, :]], 1)
            conv = tf.nn.conv1d(bottom_pad, filt, stride_size, padding='VALID')
        else:
            tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def circular_conv_1d_complex(bottom, filter_size, in_channels, out_channels,
                             name, stride_size=1, biases=False, bias_scale=1.,
                             FFT=False):
    with tf.variable_scope(name, reuse=None):
        filt_re, conv_biases_re = get_conv_var1d(filter_size, in_channels, out_channels,
                                                 name="real_", biases=biases, bias_scale=bias_scale)
        filt_im, conv_biases_im = get_conv_var1d(filter_size, in_channels, out_channels,
                                                 name="imag_", biases=biases, bias_scale=bias_scale)
        if not FFT:
            # bottom shape [None, Lx, channels]
            # pad_size = filter_size - 1
            bottom_pad = tf.concat([bottom, bottom[:, :filter_size-1, :]], 1)
            bottom_pad_re = tf.real(bottom_pad)
            bottom_pad_im = tf.real(bottom_pad)
            conv_re = (tf.nn.conv1d(bottom_pad_re, filt_re, stride_size, padding='VALID') -
                       tf.nn.conv1d(bottom_pad_im, filt_im, stride_size, padding='VALID'))
            conv_im = (tf.nn.conv1d(bottom_pad_im, filt_re, stride_size, padding='VALID') +
                       tf.nn.conv1d(bottom_pad_re, filt_im, stride_size, padding='VALID'))
            conv = tf.complex(conv_re, conv_im)
        else:
            filt = tf.complex(filt_re, filt_im)
            tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            conv_biases = tf.complex(conv_biases_re, conv_biases_im)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def circular_conv_2d(bottom, filter_size, in_channels, out_channels,
                     name, stride_size=1, biases=False, bias_scale=1.,
                     FFT=False):
    '''
    FFT can be used instead of circular convolution. Although, the pad and conv
    approach is relatively slow comparing to ordinary conv operation, it is still
    much faster than FFT approach generally, since the dimension of the filter is small.
    '''
    with tf.variable_scope(name, reuse=None):
        filt, conv_biases = get_conv_var2d(filter_size, in_channels, out_channels,
                                           biases=biases, bias_scale=bias_scale)
        if not FFT:
            # bottom shape [None, Lx, Ly, channels]
            # pad_size = filter_size - 1
            bottom_pad_x = tf.concat([bottom, bottom[:, :filter_size-1, :, :]], 1)
            bottom_pad_xy = tf.concat([bottom_pad_x, bottom_pad_x[:, :, :filter_size-1, :]], 2)
            stride_list = [1, stride_size, stride_size, 1]
            conv = tf.nn.conv2d(bottom_pad_xy, filt, stride_list, padding='VALID')
        else:
            raise NotImplementedError
            # tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            # tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            # tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            # tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            # conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def circular_conv_2d_complex(bottom, filter_size, in_channels, out_channels,
                             name, stride_size=1, biases=False, bias_scale=1.,
                             FFT=False):
    with tf.variable_scope(name, reuse=None):
        filt_re, conv_biases_re = get_conv_var2d(filter_size, in_channels, out_channels,
                                                 name="real_", biases=biases, bias_scale=bias_scale)
        filt_im, conv_biases_im = get_conv_var2d(filter_size, in_channels, out_channels,
                                                 name="imag_", biases=biases, bias_scale=bias_scale)
        if not FFT:
            # bottom shape [None, Lx, Ly, channels]
            # pad_size = filter_size - 1
            bottom_pad_x = tf.concat([bottom, bottom[:, :filter_size-1, :, :]], 1)
            bottom_pad_xy = tf.concat([bottom_pad_x, bottom_pad_x[:, :, :filter_size-1, :]], 2)
            bottom_pad_re = tf.real(bottom_pad_xy)
            bottom_pad_im = tf.real(bottom_pad_xy)
            stride_list = [1, stride_size, stride_size, 1]
            conv_re = (tf.nn.conv2d(bottom_pad_re, filt_re, stride_list, padding='VALID') -
                       tf.nn.conv2d(bottom_pad_im, filt_im, stride_list, padding='VALID'))
            conv_im = (tf.nn.conv2d(bottom_pad_im, filt_re, stride_list, padding='VALID') +
                       tf.nn.conv2d(bottom_pad_re, filt_im, stride_list, padding='VALID'))
            conv = tf.complex(conv_re, conv_im)
        else:
            raise NotImplementedError
            # filt = tf.complex(filt_re, filt_im)
            # tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            # tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            # tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            # tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            # conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            conv_biases = tf.complex(conv_biases_re, conv_biases_im)
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
                   biases=False, dtype=tf.float32, bias_scale=1.):
    if dtype == tf.complex64:
        raise NotImplementedError
        # tensorflow optimizer does not support complex type
    else:
        pass

    # initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0, 0.1)
    initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0,
                                        np.sqrt(2. / (filter_size * (in_channels + out_channels))))
    filters = get_var(initial_value, name + "weights", dtype=dtype)

    if not biases:
        return filters, None
    else:
        initial_value = tf.truncated_normal([out_channels], .0, .001 * bias_scale)
        biases = get_var(initial_value, name + "biases", dtype=dtype)
        return filters, biases


def get_conv_var2d(filter_size, in_channels, out_channels, name="",
                   biases=False, dtype=tf.float32, bias_scale=1.):
    if dtype == tf.complex64:
        raise NotImplementedError
        # tensorflow optimizer does not support complex type
    else:
        pass

    # initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
    initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0,
                                        np.sqrt(2. / (filter_size * filter_size * in_channels )))
                                        # np.sqrt(2. / (filter_size*filter_size*(in_channels+out_channels))))
                                        # Xavier init
    filters = get_var(initial_value, name + "weights", dtype=dtype)

    if not biases:
        return filters, None
    else:
        initial_value = tf.truncated_normal([out_channels], .0, .001 * bias_scale)
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
    # Xavier init
    # initial_value = tf.random_normal([in_size, out_size], stddev=np.sqrt(2./(in_size+out_size)))
    # He (MSAR) init
    initial_value = tf.random_normal([in_size, out_size], stddev=np.sqrt(2./(in_size)))
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


def bottleneck_residual(x, in_channel, out_channel, name,
                        stride_size=2):
    with tf.variable_scope(name, reuse=None):
        # Identity shortcut
        if in_channel == out_channel:
            shortcut = x
            x = self.conv_layer2d(x, 1, in_channel, out_channel/4, "conv1")
            # conv projection shortcut
        else:
            shortcut = x
            shortcut = self.conv_layer2d(shortcut, 1, in_channel,
                                         out_channel, "shortcut",
                                         stride_size=stride_size)
            shortcut = self.batch_norm(shortcut, phase=self.bn_is_training,
                                       scope='shortcut/bn')
            x = self.conv_layer2d(x, 1, in_channel, out_channel/4, "conv1",
                                  stride_size=stride_size)

        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
        x = tf.nn.relu(x)
        x = self.conv_layer2d(x, 3, out_channel/4, out_channel/4, "conv2")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn2')
        x = tf.nn.relu(x)
        x = self.conv_layer2d(x, 1, out_channel/4, out_channel, "conv3")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn3')
        x += shortcut
        x = tf.nn.relu(x)

    return x

def residual_block(x, num_channel, name, stride_size=1, activation=tf.nn.relu, bn_is_training=True):
    '''
    This is an implementation of standard residual networks with BN after convolution.

    Args:
        inputs: x, a tensor of size [num_data, lx, ly, num_channels]
        num_channel: number of channel for both input and output
        name: name of this residual block
        stride size:
    '''
    in_channel = num_channel
    out_channel = num_channel
    with tf.variable_scope(name, reuse=None):
        # conv projection shortcut
        shortcut = x
        x = circular_conv_2d(x, 3, in_channel, out_channel, "conv1",
                             stride_size=stride_size, biases=True)
        x = batch_norm(x, phase=bn_is_training, scope='bn1')
        x = activation(x)
        x = circular_conv_2d(x, 3, in_channel, out_channel, "conv2",
                             stride_size=stride_size, biases=True)
        x = batch_norm(x, phase=bn_is_training, scope='bn2')
        x += shortcut
        x = activation(x)

    return x

def get_jastrow_var(n_body, dimension, name="", dtype=tf.float32, scale=1.):
    '''
    return a 2-body jastrow factor.
    If n_body = 2: J(dim,dim)
    '''
    if n_body != 2:
        raise NotImplementedError

    j_factor_size = [dimension] * n_body
    initial_value = tf.random_uniform(j_factor_size, minval=-.1 * scale,
                                      maxval=.1 * scale, dtype=dtype)
    weights = get_var(initial_value, name + "weights", dtype=dtype)
    weights_upper = tf.matrix_band_part(weights, 0, -1)
    weights_symm = 0.5 * (weights_upper + tf.transpose(weights_upper))
    return weights_symm

def jastrow_2d_amp(config_array, Lx, Ly, local_d, name, sym=False):
    with tf.variable_scope(name, reuse=None):
        total_dim = Lx * Ly * local_d
        # get symmetry weights matrix, (total_dim x total_dim )
        weights_symm_re = get_jastrow_var(2, total_dim, name="real_", dtype=tf.float32)
        weights_symm_im = get_jastrow_var(2, total_dim, name="imag_", dtype=tf.float32, scale=15)

        config_vector = tf.reshape(config_array, [-1, total_dim])
        C2_array = tf.einsum('ij,ik->ijk', config_vector, config_vector)
        C2_array = tf.multiply(tf.complex(C2_array, tf.zeros_like(C2_array)),
                               tf.complex(weights_symm_re, weights_symm_im))
        amp_array = tf.reduce_sum(C2_array, axis=[1, 2], keep_dims=False)

    return tf.exp(amp_array)


def jacobian(y, x):
    y_flat = tf.reshape(y, (-1,))
    jacobian_flat = tf.stack(
        [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
    return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))
