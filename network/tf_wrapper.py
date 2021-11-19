import tensorflow as tf
from functools import reduce
import numpy as np
from . import mask
from . import layers


# Inverse update ops will be run every _INVERT_EVRY iterations.
_INVERT_EVERY = 10
# Covariance matrices will be update  _COV_UPDATE_EVERY iterations.
_COV_UPDATE_EVERY = 1
# Displays loss every _REPORT_EVERY iterations.
_REPORT_EVERY = 10


def select_optimizer(optimizer,
                     learning_rate,
                     momentum=0,
                     var_list=None,
                     layer_collection=None):
    if optimizer == 'Adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
        # epsilon=1e-3) #previous 1e-8
    elif optimizer == 'Mom':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=momentum)
    elif optimizer == 'RMSprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                         epsilon=1e-6)  # previous 1e-1
    elif optimizer == 'GD':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer == 'Adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                          epsilon=1e-6)
    elif optimizer == 'GGT':
        return tf.contrib.opt.GGTOptimizer(learning_rate=learning_rate,
                                           window=128)
    elif optimizer == 'KFAC':
        # The following argument is for setting up kfac optimizer
        import sys
        # We append the relative path to kfac cloned directory
        sys.path.append("../../kfac")
        import kfac

        # return kfac.KfacOptimizer(
        #     learning_rate=learning_rate,
        #     var_list=var_list,
        #     cov_ema_decay=0.95,
        #     damping=0.001,
        #     layer_collection=layer_collection,
        #     estimation_mode="empirical",
        #     placement_strategy="round_robin",
        #     # cov_devices=[device],
        #     # inv_devices=[device],
        #     momentum=0.9)
        return kfac.PeriodicInvCovUpdateKfacOpt(
            invert_every=_INVERT_EVERY,
            cov_update_every=_COV_UPDATE_EVERY,
            learning_rate=learning_rate,
            cov_ema_decay=0.95,
            damping=0.1,
            layer_collection=layer_collection,
            placement_strategy="round_robin",
            # cov_devices=[device],
            # inv_devices=[device],
            # trans_devices=[device],
            var_list=var_list,
            estimation_mode="empirical",
            momentum=0.9)
    else:
        raise


def select_activation(activation):
    if activation == 'softplus2':
        return softplus2
    elif activation == 'softplus':
        return softplus
    elif activation == 'complex_relu':
        return complex_relu
    elif activation == 'c_relu':
        return c_relu
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'c_elu':
        return c_elu
    elif activation == 'elu':
        return tf.nn.elu
    elif activation == 'tanh':
        return tf.tanh
    elif activation == 'sin':
        return tf.sin
    elif activation == 'selu':
        return tf.nn.selu
    elif activation == 'lncosh':
        return logcosh
    elif activation == 'linear':
        return linear
    elif activation == 'poly':
        return polynomial
    else:
        raise NotImplementedError


def linear(x):
    return x


def logcosh(x):
    return tf.math.log(tf.math.cosh(x))


def polynomial(x):
    return (x**2)/2. - (x**4)/12. + (x**6)/45.


def leaky_relu(x):
    return tf.maximum(0.01 * x, x)


def softplus(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x)))


def softplus2(x):
    return tf.log(tf.add(tf.ones_like(x), tf.exp(x)) / 2.)


def complex_relu(x):
    re = tf.real(x)
    # im = tf.imag(x)
    return tf.where(tf.greater(re, tf.zeros_like(re)), x,
                    tf.complex(tf.nn.relu(re), tf.zeros_like(re)))
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
    return tf.nn.max_pool(x,
                          ksize=[1, kernel_size, 1, 1],
                          strides=[1, stride_size, 1, 1],
                          padding=padding,
                          name=name)


def max_pool2d(x, name, kernel_size=2, stride_size=2, padding='SAME'):
    return tf.nn.max_pool(x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding,
                          name=name)


def avg_pool1d(x, name, kernel_size, stride_size=2, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, kernel_size, 1, 1],
                          strides=[1, stride_size, 1, 1],
                          padding=padding,
                          name=name)


def avg_pool2d(x, name, kernel_size, stride_size=2, padding='SAME'):
    return tf.nn.avg_pool(x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding,
                          name=name)


def kron(A, B, rankA, dimA, dimB, swap=False):
    """
    Returns Kronecker product of two square matrices.

    Args:
        A: tf tensor of shape (..., dim1, dim1)
        B: tf tensor of shape (dim2, dim2)
        A may contain batch dimensions
        Not assuming complex or real-valued.
        A, B have to have the same dtype

    Returns:
        tf tensor of shape (..., dim1 * dim2, dim1 * dim2),
        kronecker product of two matrices
    """

    if not swap:
        AB = tf.transpose(tf.tensordot(A, B, axes=0), list(range(rankA-2)) +
                          [rankA-2+0, rankA-2+2, rankA-2+1, rankA-2+3])
        shape = AB.get_shape().as_list()  # get the shape of each dimention
        return tf.reshape(AB, [-1] + shape[1:rankA-2] + [dimA * dimB, dimA * dimB])
    else:
        AB = tf.transpose(tf.tensordot(A, B, axes=0), list(range(rankA-2)) +
                          [rankA-2+2, rankA-2+0, rankA-2+3, rankA-2+1])
        shape = AB.get_shape().as_list()  # get the shape of each dimention
        return tf.reshape(AB, [-1] + shape[1:rankA-2] + [dimA * dimB, dimA * dimB])


def gen_2_qubit_gate(x):
    '''
    [NOT TESTED BACKPROP BUG FREE]
    Input:
        real-valued tensor with dimension [..., N]
        where N > 15, the rest is abandoned

    Return:
        complex-valued tensor with dimension [..., 4, 4]

    Goal:
    circuit += one_qubit_unitary(bits[0], symbols[0:3])
    circuit += one_qubit_unitary(bits[1], symbols[3:6])
    circuit += [cirq.ZZ(*bits)**symbols[6]]
    circuit += [cirq.YY(*bits)**symbols[7]]
    circuit += [cirq.XX(*bits)**symbols[8]]
    circuit += one_qubit_unitary(bits[0], symbols[9:12])
    circuit += one_qubit_unitary(bits[1], symbols[12:])
    '''
    np_X = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    np_Y = np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128)
    np_Z = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    np_I = np.array([[1., 0.], [0., 1.]], dtype=np.complex128)
    tf_X = tf.constant(np_X)
    tf_Y = tf.constant(np_Y)
    tf_Z = tf.constant(np_Z)
    tf_I = tf.constant(np_I)
    tf_XX = tf.constant(np.kron(np_X, np_X))
    tf_YY = tf.constant(np.kron(np_Y, np_Y))
    tf_ZZ = tf.constant(np.kron(np_Z, np_Z))

    sig_mat = tf.stack([tf_X, tf_Y, tf_Z])  # [3, 2, 2]
    sig_sig_mat = tf.stack([tf_XX, tf_YY, tf_ZZ])  # [3, 4, 4]

    x = tf.Print(x, [x[:, -1:, 0:8]], message='0-8')
    x = tf.Print(x, [x[:, -1:, 8:]], message='8-')
    x = tf.cast(x, dtype=tf.complex128)
    rot1 = tf.linalg.expm(1.j * tf.tensordot(x[:, :, 0:3], sig_mat, axes=[[-1], [0]]))  # N, L, 2, 2
    rot2 = tf.linalg.expm(1.j * tf.tensordot(x[:, :, 3:6], sig_mat, axes=[[-1], [0]]))
    rot3 = tf.linalg.expm(1.j * tf.tensordot(x[:, :, 6:9], sig_sig_mat, axes=[[-1], [0]]))
    rot4 = tf.linalg.expm(1.j * tf.tensordot(x[:, :, 9:12], sig_mat, axes=[[-1], [0]]))
    rot5 = tf.linalg.expm(1.j * tf.tensordot(x[:, :, 12:15], sig_mat, axes=[[-1], [0]]))
    # rot1 = tf.Print(rot1, [tf.real(rot1[0, -1, :, :])], message='rotation real  ')
    # rot1 = tf.Print(rot1, [tf.imag(rot1[0, -1, :, :])], message='rotation imag  ')
    rot1 = kron(rot1, tf_I, rankA=4, dimA=2, dimB=2, swap=False)
    rot2 = kron(rot2, tf_I, rankA=4, dimA=2, dimB=2, swap=True)
    rot4 = kron(rot4, tf_I, rankA=4, dimA=2, dimB=2, swap=False)
    rot5 = kron(rot5, tf_I, rankA=4, dimA=2, dimB=2, swap=True)

    iter_mat = rot5
    for rot in [rot4, rot3, rot2, rot1]:
        iter_mat = tf.linalg.matmul(iter_mat, rot)

    return iter_mat


def batch_norm(bottom, phase, scope='bn'):
    return tf.contrib.layers.batch_norm(bottom,
                                        center=True,
                                        scale=True,
                                        is_training=phase,
                                        scope=scope,
                                        decay=0.995,
                                        reuse=tf.AUTO_REUSE)


def batch_norm_new(bottom, phase, scope_name='bn'):
    _BATCH_NORM_DECAY = 0.997
    _BATCH_NORM_EPSILON = 1e-5
    return tf.layers.batch_normalization(
        inputs=bottom,
        axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=phase,
        fused=True,
        name=scope_name)


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)  # N, L_p, C
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])  # N, L_p_p, C
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])  # N * (L_p_p)/d, d, C
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])  # d, N * (L_p_p)/d, C
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])  # N*d, (L_p_p/d), C


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)  # N*d, (L_p_p/d), C
        prepared = tf.reshape(value, [dilation, -1, shape[2]])  # d, N * (L_p_p)/d, C
        transposed = tf.transpose(prepared, perm=[1, 0, 2])  # N * (L_p_p)/d, d, C
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])  # N, L_p_p, C


def conv_layer1d(bottom,
                 filter_size,
                 in_channels,
                 out_channels,
                 name,
                 stride_size=1,
                 padding='SAME',
                 dilations=None,
                 biases=True,
                 dtype=tf.float64,
                 weight_normalization=False,
                 layer_collection=None,
                 registered=False
                 ):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt, conv_biases = get_conv_var1d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           dtype=dtype
                                           )
        stride_list = [1, stride_size, 1]
        if dilations is None or dilations == 1:
            conv = tf.nn.conv1d(bottom, filt, stride_list, padding=padding)
        else:
            assert stride_size == 1
            assert padding == 'VALID'
            transformed = time_to_batch(bottom, dilations)
            conv = tf.nn.conv1d(transformed, filt, stride_list,
                                padding='VALID')
            restored = batch_to_time(conv, dilations)
            # Remove excess elements at the end. We assume the input is padded with
            # the causal padding, i.e. (filter_size - 1) * dilations
            out_width = tf.shape(bottom)[1] - (filter_size - 1) * dilations
            conv = tf.slice(restored,
                            [0, 0, 0],
                            [-1, out_width, -1])

        if biases:
            conv = tf.nn.bias_add(conv, conv_biases)
            params = [filt, conv_biases]
        else:
            params = filt

        if (layer_collection is not None) and (registered == False):
            layer_collection.register_conv1d(params, stride_list, padding,
                                             bottom, conv)

        return conv


def masked_conv_layer1d(bottom,
                        filter_size,
                        in_channels,
                        out_channels,
                        mask_type,
                        name,
                        stride_size=1,
                        padding='SAME',
                        biases=True,
                        dtype=tf.float64,
                        weight_normalization=False,
                        layer_collection=None,
                        registered=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        np_mask = mask.gen_1d_conv_mask(mask_type, filter_size, in_channels,
                                        out_channels)
        tf_mask = tf.constant(np_mask, dtype=dtype)

        filt, conv_biases = get_conv_var1d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           dtype=dtype)
        stride_list = [1, stride_size, 1]
        if not weight_normalization:
            conv = tf.nn.conv1d(bottom,
                                filt * tf_mask,
                                stride_list,
                                padding=padding)
        else:
            g = tf.get_variable('g', dtype=dtype,
                                initializer=tf.math.reduce_sum(tf.math.square(filt), axis=[0, 1]),
                                trainable=True)
            repara_filt = tf.reshape(g, [1, 1, out_channels]) * tf.nn.l2_normalize(filt*tf_mask, [0, 1])
            conv = tf.nn.conv1d(bottom,
                                repara_filt,
                                stride_list,
                                padding=padding)

        if biases:
            conv = tf.nn.bias_add(conv, conv_biases)
            params = [filt, conv_biases]
        else:
            params = filt

        if (layer_collection is not None) and (registered == False):
            layer_collection.register_conv1d(params, stride_list, padding,
                                             bottom, conv)

        return conv


def circular_conv_1d(bottom,
                     filter_size,
                     in_channels,
                     out_channels,
                     name,
                     stride_size=1,
                     biases=False,
                     bias_scale=1.,
                     FFT=False,
                     dtype=tf.float64):
    '''
    FFT can be used instead of circular convolution. Although, the pad and conv
    approach is relatively slow comparing to ordinary conv operation, it is still
    much faster than FFT approach generally, since the dimension of the filter is small.
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt, conv_biases = get_conv_var1d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           bias_scale=bias_scale,
                                           dtype=dtype)
        if not FFT:
            # bottom shape [None, Lx, channels]
            # pad_size = filter_size - 1
            bottom_pad = tf.concat([bottom, bottom[:, :filter_size - 1, :]], 1)
            conv = tf.nn.conv1d(bottom_pad, filt, stride_size, padding='VALID')
        else:
            tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft),
                                  tf.conj(tf_W_fft))
            tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def circular_conv_1d_complex(bottom,
                             filter_size,
                             in_channels,
                             out_channels,
                             name,
                             stride_size=1,
                             biases=False,
                             bias_scale=1.,
                             FFT=False,
                             dtype=tf.complex128):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt_re, conv_biases_re = get_conv_var1d(filter_size,
                                                 in_channels,
                                                 out_channels,
                                                 name="real_",
                                                 biases=biases,
                                                 bias_scale=bias_scale,
                                                 dtype=dtype)
        filt_im, conv_biases_im = get_conv_var1d(filter_size,
                                                 in_channels,
                                                 out_channels,
                                                 name="imag_",
                                                 biases=biases,
                                                 bias_scale=bias_scale,
                                                 dtype=dtype)
        if not FFT:
            # bottom shape [None, Lx, channels]
            # pad_size = filter_size - 1
            bottom_pad = tf.concat([bottom, bottom[:, :filter_size - 1, :]], 1)
            bottom_pad_re = tf.real(bottom_pad)
            bottom_pad_im = tf.imag(bottom_pad)
            conv_re = (
                tf.nn.conv1d(
                    bottom_pad_re, filt_re, stride_size, padding='VALID') -
                tf.nn.conv1d(
                    bottom_pad_im, filt_im, stride_size, padding='VALID'))
            conv_im = (
                tf.nn.conv1d(
                    bottom_pad_im, filt_re, stride_size, padding='VALID') +
                tf.nn.conv1d(
                    bottom_pad_re, filt_im, stride_size, padding='VALID'))
            conv = tf.complex(conv_re, conv_im)
        else:
            filt = tf.complex(filt_re, filt_im)
            tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft),
                                  tf.conj(tf_W_fft))
            tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            conv = tf.real(tf_XW)

        if not biases:
            return conv
        else:
            conv_biases = tf.complex(conv_biases_re, conv_biases_im)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias


def circular_conv_2d(bottom,
                     filter_size,
                     in_channels,
                     out_channels,
                     name,
                     stride_size=1,
                     biases=False,
                     bias_scale=1.,
                     FFT=False,
                     layer_collection=None,
                     registered=False,
                     dtype=tf.float64):
    '''
    FFT can be used instead of circular convolution. Although, the pad and conv
    approach is relatively slow comparing to ordinary conv operation, it is still
    much faster than FFT approach generally, since the dimension of the filter is small.
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt, conv_biases = get_conv_var2d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           bias_scale=bias_scale,
                                           dtype=dtype)
        if not FFT:
            # bottom shape [None, Lx, Ly, channels]
            # pad_size = filter_size - 1
            bottom_pad_x = tf.concat(
                [bottom, bottom[:, :filter_size - 1, :, :]], 1)
            bottom_pad_xy = tf.concat(
                [bottom_pad_x, bottom_pad_x[:, :, :filter_size - 1, :]], 2)
            stride_list = [1, stride_size, stride_size, 1]
            conv = tf.nn.conv2d(bottom_pad_xy,
                                filt,
                                stride_list,
                                padding='VALID')

            if biases:
                conv = tf.nn.bias_add(conv, conv_biases)
                params = [filt, conv_biases]
            else:
                params = filt

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_conv2d(params, stride_list, 'VALID',
                                                 bottom_pad_xy, conv)

            return conv

        else:
            raise NotImplementedError
            # tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            # tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            # tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            # tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            # conv = tf.real(tf_XW)
            #
            # Add biases


def circular_conv_2d_complex(bottom,
                             filter_size,
                             in_channels,
                             out_channels,
                             name,
                             stride_size=1,
                             biases=False,
                             bias_scale=1.,
                             FFT=False,
                             layer_collection=None,
                             registered=False,
                             dtype=tf.complex128):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt_re, conv_biases_re = get_conv_var2d(filter_size,
                                                 in_channels,
                                                 out_channels,
                                                 name="real_",
                                                 biases=biases,
                                                 bias_scale=bias_scale,
                                                 dtype=dtype)
        filt_im, conv_biases_im = get_conv_var2d(filter_size,
                                                 in_channels,
                                                 out_channels,
                                                 name="imag_",
                                                 biases=biases,
                                                 bias_scale=bias_scale,
                                                 dtype=dtype)
        if not FFT:
            # bottom shape [None, Lx, Ly, channels]
            # pad_size = filter_size - 1
            bottom_pad_x = tf.concat(
                [bottom, bottom[:, :filter_size - 1, :, :]], 1)
            bottom_pad_xy = tf.concat(
                [bottom_pad_x, bottom_pad_x[:, :, :filter_size - 1, :]], 2)
            bottom_pad_re = tf.real(bottom_pad_xy)
            bottom_pad_im = tf.imag(bottom_pad_xy)
            stride_list = [1, stride_size, stride_size, 1]

            conv_RR = tf.nn.conv2d(bottom_pad_re,
                                   filt_re,
                                   stride_list,
                                   padding='VALID')
            conv_II = tf.nn.conv2d(bottom_pad_im,
                                   filt_im,
                                   stride_list,
                                   padding='VALID')
            conv_IR = tf.nn.conv2d(bottom_pad_im,
                                   filt_re,
                                   stride_list,
                                   padding='VALID')
            conv_RI = tf.nn.conv2d(bottom_pad_re,
                                   filt_im,
                                   stride_list,
                                   padding='VALID')

            if not biases:
                params_RR = [filt_re]
                params_II = [filt_im]
                params_IR = [filt_re]
                params_RI = [filt_im]
            else:
                conv_RR = tf.nn.bias_add(conv_RR, conv_biases_re)
                conv_RI = tf.nn.bias_add(conv_RI, conv_biases_im)
                params_RR = [filt_re, conv_biases_re]
                params_II = [filt_im]
                params_IR = [filt_re]
                params_RI = [filt_im, conv_biases_im]

            conv_re = conv_RR - conv_II
            conv_im = conv_IR + conv_RI
            conv = tf.complex(conv_re, conv_im)

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_conv2d(params_RR, stride_list,
                                                 'VALID', bottom_pad_re,
                                                 conv_RR)
                layer_collection.register_conv2d(params_II, stride_list,
                                                 'VALID', bottom_pad_im,
                                                 conv_II)
                layer_collection.register_conv2d(params_IR, stride_list,
                                                 'VALID', bottom_pad_im,
                                                 conv_IR)
                layer_collection.register_conv2d(params_RI, stride_list,
                                                 'VALID', bottom_pad_re,
                                                 conv_RI)

            return conv

        else:
            raise NotImplementedError
            # filt = tf.complex(filt_re, filt_im)
            # tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', bottom), 0.))
            # tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', filt), 0.))
            # tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
            # tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))
            # conv = tf.real(tf_XW)


def conv_layer2d(bottom,
                 filter_size,
                 in_channels,
                 out_channels,
                 name,
                 stride_size=1,
                 padding='SAME',
                 biases=True,
                 dtype=tf.float64,
                 weight_normalization=False,
                 layer_collection=None,
                 registered=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        filt, conv_biases = get_conv_var2d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           dtype=dtype)
        stride_list = [1, stride_size, stride_size, 1]
        if not weight_normalization:
            conv = tf.nn.conv2d(bottom, filt, stride_list, padding=padding)
        else:
            g = tf.get_variable('g', dtype=dtype,
                                initializer=tf.math.reduce_sum(tf.math.square(filt), axis=[0, 1, 2]),
                                trainable=True)
            repara_filt = tf.reshape(g, [1, 1, 1, out_channels]) * tf.nn.l2_normalize(filt, [0, 1, 2])
            conv = tf.nn.conv2d(bottom, repara_filt, stride_list, padding=padding)

        if biases:
            conv = tf.nn.bias_add(conv, conv_biases)
            params = [filt, conv_biases]
        else:
            params = filt

        if (layer_collection is not None) and (registered == False):
            layer_collection.register_conv2d(params, stride_list, padding,
                                             bottom, conv)

        return conv


def masked_conv_layer2d(bottom,
                        filter_size,
                        in_channels,
                        out_channels,
                        mask_type,
                        name,
                        stride_size=1,
                        padding='SAME',
                        biases=True,
                        dtype=tf.float64,
                        weight_normalization=False,
                        layer_collection=None,
                        registered=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        np_mask = mask.gen_2d_conv_mask(mask_type, filter_size, in_channels,
                                        out_channels)
        tf_mask = tf.constant(np_mask, dtype=dtype)

        filt, conv_biases = get_conv_var2d(filter_size,
                                           in_channels,
                                           out_channels,
                                           biases=biases,
                                           dtype=dtype)
        stride_list = [1, stride_size, stride_size, 1]
        if not weight_normalization:
            conv = tf.nn.conv2d(bottom,
                                filt * tf_mask,
                                stride_list,
                                padding=padding)
        else:
            g = tf.get_variable('g', dtype=dtype,
                                initializer=tf.math.reduce_sum(tf.math.square(filt), axis=[0, 1, 2]),
                                trainable=True)
            repara_filt = tf.reshape(g, [1, 1, 1, out_channels]) * tf.nn.l2_normalize(filt*tf_mask, [0, 1, 2])
            conv = tf.nn.conv2d(bottom,
                                repara_filt,
                                stride_list,
                                padding=padding)

        if biases:
            conv = tf.nn.bias_add(conv, conv_biases)
            params = [filt, conv_biases]
        else:
            params = filt

        if (layer_collection is not None) and (registered == False):
            layer_collection.register_conv2d(params, stride_list, padding,
                                             bottom, conv)

        return conv


def fc_layer(bottom,
             in_size,
             out_size,
             name,
             biases=True,
             dtype=tf.float64,
             init_style='He',
             layer_collection=None,
             registered=False,
            ):
    '''
    The if...else... below could merge.
    '''
    if dtype not in [tf.complex64, tf.complex128]:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights, biases = get_fc_var(in_size,
                                         out_size,
                                         biases=biases,
                                         dtype=dtype,
                                         init_style=init_style
                                        )
            x = tf.reshape(bottom, [-1, in_size])
            if biases:
                fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
                params = [weights, biases]
            else:
                fc = tf.matmul(x, weights)
                params = weights

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_fully_connected(params, x, fc)

    else:
        part_dtype = {tf.complex64: tf.float32, tf.complex128: tf.float64}
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            complex_weights, complex_biases = get_fc_var(in_size,
                                                         out_size,
                                                         biases=biases,
                                                         dtype=dtype)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, complex_weights)
            if biases:
                fc = tf.nn.bias_add(fc, complex_biases)
                params = [complex_weights, complex_biases]
            else:
                params = complex_weights
                pass

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_fully_connected(params, x, fc)

    return fc


def masked_fc_layer(bottom,
                    in_size,
                    out_size,
                    name,
                    ordering,
                    mask_type,
                    biases=True,
                    dtype=tf.float64,
                    layer_collection=None,
                    registered=False):
    '''
    implementing the masked fully connected layer.
    The in_size and the out_size should be a multiple of size of the ordering.
    in_size / |ordering| = # in_hidden
    out_size / |ordering| = # out_hidden

    https://stackoverflow.com/questions/44915379/arbitrary-filters-for-conv2d-as-opposed-to-rectangular
    '''
    N = ordering.size
    assert in_size % N == 0
    assert out_size % N == 0
    in_hidden = in_size // N
    out_hidden = out_size // N

    np_dtype = {
        tf.float32: np.float32,
        tf.float64: np.float64,
        tf.complex64: np.complex64,
        tf.complex128: np.complex128
    }
    np_mask = mask.gen_fc_mask(ordering,
                               mask_type=mask_type,
                               dtype=np_dtype[dtype],
                               in_hidden=in_hidden,
                               out_hidden=out_hidden)
    tf_mask = tf.constant(np_mask, dtype=dtype)

    if dtype not in [tf.complex64, tf.complex128]:
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights, biases = get_fc_var(in_size,
                                         out_size,
                                         biases=biases,
                                         dtype=dtype)
            x = tf.reshape(bottom, [-1, in_size])
            if biases:
                fc = tf.nn.bias_add(tf.matmul(x, weights * tf_mask), biases)
                params = [weights, biases]
            else:
                fc = tf.matmul(x, weights)
                params = weights

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_fully_connected(params, x, fc)

    else:
        part_dtype = {tf.complex64: tf.float32, tf.complex128: tf.float64}
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            complex_weights, complex_biases = get_fc_var(in_size,
                                                         out_size,
                                                         biases=biases,
                                                         dtype=dtype)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.matmul(x, complex_weights * tf_mask)
            if biases:
                fc = tf.nn.bias_add(fc, complex_biases)
                params = [complex_weights, complex_biases]
            else:
                params = complex_weights
                pass

            if (layer_collection is not None) and (registered == False):
                layer_collection.register_fully_connected(params, x, fc)

    return fc


def get_conv_var1d(filter_size,
                   in_channels,
                   out_channels,
                   name="",
                   biases=False,
                   dtype=tf.float32,
                   bias_scale=1.):
    if dtype in [tf.complex64, tf.complex128]:
        # tensorflow optimizer does not support complex type
        # So we init with two sets of real variables
        if dtype == tf.complex64:
            part_dtype = tf.float32
        else:
            part_dtype = tf.float64

        u = tf.random_uniform([filter_size, in_channels, out_channels],
                              minval=0,
                              maxval=1.)
        sigma = np.sqrt(2. / (filter_size * in_channels))
        w_mag = sigma * tf.sqrt(-2. * tf.log(1. - u))
        theta = tf.random_uniform([filter_size, in_channels, out_channels],
                                  minval=0,
                                  maxval=2. * np.pi)
        init_w_re = w_mag * tf.cos(theta)
        init_w_im = w_mag * tf.sin(theta)
        real_weights = get_var(init_w_re,
                               name + "real_weights",
                               dtype=part_dtype)
        imag_weights = get_var(init_w_im,
                               name + "imag_weights",
                               dtype=part_dtype)
        weights = tf.complex(real_weights, imag_weights)
        if biases:
            re_initial_value = tf.zeros(out_channels, dtype=part_dtype)
            real_biases = get_var(re_initial_value,
                                  name + "real_biases",
                                  dtype=part_dtype)
            im_initial_value = tf.random_uniform([out_channels],
                                                 minval=0,
                                                 maxval=2. * np.pi)
            imag_biases = get_var(im_initial_value,
                                  name + "imag_biases",
                                  dtype=part_dtype)
            biases = tf.complex(real_biases, imag_biases)
            return weights, biases
        else:
            return weights, None

    else:
        # initial_value = tf.truncated_normal([filter_size, in_channels, out_channels], 0.0, 0.1)
        initial_value = tf.truncated_normal(
            [filter_size, in_channels, out_channels],
            0.0,
            np.sqrt(2. / (filter_size * (in_channels + out_channels))),
            dtype=dtype)
        filters = get_var(initial_value, name + "weights", dtype=dtype)

        if not biases:
            return filters, None
        else:
            initial_value = tf.truncated_normal([out_channels],
                                                .0,
                                                .001 * bias_scale,
                                                dtype=dtype)
            biases = get_var(initial_value, name + "biases", dtype=dtype)
            return filters, biases


def get_conv_var2d(filter_size,
                   in_channels,
                   out_channels,
                   name="",
                   biases=False,
                   dtype=tf.float64,
                   bias_scale=1.):
    if dtype in [tf.complex64, tf.complex128]:
        # tensorflow optimizer does not support complex type
        # So we init with two sets of real variables
        if dtype == tf.complex64:
            part_dtype = tf.float32
        else:
            part_dtype = tf.float64

        u = tf.random_uniform(
            [filter_size, filter_size, in_channels, out_channels],
            minval=0,
            maxval=1.,
            dtype=part_dtype)
        sigma = np.sqrt(2. / (filter_size * filter_size * in_channels))
        w_mag = sigma * tf.sqrt(-2. * tf.log(1. - u))
        theta = tf.random_uniform(
            [filter_size, filter_size, in_channels, out_channels],
            minval=0,
            maxval=2. * np.pi,
            dtype=part_dtype)
        init_w_re = w_mag * tf.cos(theta)
        init_w_im = w_mag * tf.sin(theta)
        real_weights = get_var(init_w_re,
                               name + "real_weights",
                               dtype=part_dtype)
        imag_weights = get_var(init_w_im,
                               name + "imag_weights",
                               dtype=part_dtype)
        weights = tf.complex(real_weights, imag_weights)
        if biases:
            re_initial_value = tf.zeros(out_channels, dtype=part_dtype)
            real_biases = get_var(re_initial_value,
                                  name + "real_biases",
                                  dtype=part_dtype)
            im_initial_value = tf.random_uniform([out_channels],
                                                 minval=0,
                                                 maxval=2. * np.pi)
            imag_biases = get_var(im_initial_value,
                                  name + "imag_biases",
                                  dtype=part_dtype)
            biases = tf.complex(real_biases, imag_biases)
            return weights, biases
        else:
            return weights, None

    else:
        # initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
        initial_value = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.,
            np.sqrt(2. / (filter_size * filter_size * in_channels) * 0.5),
            dtype=dtype)
        # np.sqrt(2. / (filter_size*filter_size*(in_channels+out_channels))))
        # Xavier init
        filters = get_var(initial_value, name + "weights", dtype=dtype)

        if not biases:
            return filters, None
        else:
            initial_value = tf.truncated_normal([out_channels],
                                                .0,
                                                .001 * bias_scale,
                                                dtype=dtype)
            biases = get_var(initial_value, name + "biases", dtype=dtype)
            return filters, biases


def get_fc_var(in_size, out_size, name="", biases=True, dtype=tf.float64, init_style='He'):
    if dtype in [tf.complex64, tf.complex128]:
        # tensorflow optimizer does not support complex type
        # So we init with two sets of real variables
        if dtype == tf.complex64:
            part_dtype = tf.float32
        else:
            part_dtype = tf.float64

        u = tf.random_uniform([in_size, out_size], minval=0, maxval=1., dtype=part_dtype)
        sigma = np.sqrt(2. / in_size)
        w_mag = sigma * tf.sqrt(-2. * tf.log(1. - u))
        theta = tf.random_uniform([in_size, out_size],
                                  minval=0,
                                  maxval=2. * np.pi,
                                  dtype=part_dtype)
        init_w_re = w_mag * tf.cos(theta)
        init_w_im = w_mag * tf.sin(theta)
        real_weights = get_var(init_w_re,
                               name + "real_weights",
                               dtype=part_dtype)
        imag_weights = get_var(init_w_im,
                               name + "imag_weights",
                               dtype=part_dtype)
        weights = tf.complex(real_weights, imag_weights)
        if biases:
            re_initial_value = tf.zeros(out_size, dtype=part_dtype)
            real_biases = get_var(re_initial_value,
                                  name + "real_biases",
                                  dtype=part_dtype)
            im_initial_value = tf.random_uniform([out_size],
                                                 minval=0,
                                                 maxval=2. * np.pi,
                                                 dtype=part_dtype
                                                 )
            imag_biases = get_var(im_initial_value,
                                  name + "imag_biases",
                                  dtype=part_dtype)
            biases = tf.complex(real_biases, imag_biases)
            return weights, biases
        else:
            return weights, None

    else:
        # initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1)
        if init_style == 'Xavier':
            # Xavier init
            initial_value = tf.random_normal([in_size, out_size], stddev=np.sqrt(2./(in_size+out_size)), dtype=dtype)
        elif init_style == 'He':
            # He (MSAR) init
            initial_value = tf.random_normal([in_size, out_size],
                                             stddev=np.sqrt(2. / (in_size)),
                                             dtype=dtype)

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
    var = tf.get_variable(var_name,
                          initializer=initial_value,
                          trainable=True,
                          dtype=dtype)
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


def pixel_block_sharir(x,
                       in_channel,
                       out_channel,
                       block_type,
                       name,
                       dtype,
                       filter_size=3,
                       activation=tf.nn.relu,
                       layer_collection=None,
                       registered=False,
                       residual_connection=False,
                       weight_normalization=False,
                       BN=False, bn_phase=None,
                       split_block=False):
    '''
    for starting block, input: x, output: out with two branch concat in channel dimension
    for mid block,  input x with two branch concat in channel dimension
                    output with two branch concat in channel dimension.
    for end block,  input x with two branch concat in channel dimension
                    output with 4 channel, representing spin up spin down amp = exp(r+i\theta)
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Starting block
        if block_type == 'start':
            assert out_channel % 2 == 0
            vertical_branch = x
            horizontal_branch = x
            # Should add padding, top 2 rows
            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            vertical_branch = conv_layer2d(ver_padded_x,
                                           filter_size,
                                           in_channel,
                                           out_channel // 2,
                                           name + '_ver',
                                           dtype=dtype,
                                           padding='VALID',
                                           weight_normalization=weight_normalization,
                                           layer_collection=layer_collection,
                                           registered=registered)

            # If the activation below applies here, then
            # it is applied before the down_shift concatanation step
            # This might not be what we want
            #
            # vertical_branch = activation(vertical_branch)

            # N H W C
            down_shift_v_branch = tf.pad(vertical_branch[:, :-1, :, :],
                                         [[0, 0], [1, 0], [0, 0], [0, 0]],
                                         "CONSTANT")
            horizontal_branch = tf.concat(
                [down_shift_v_branch, horizontal_branch], axis=3)

            # Should add padding, top 2 rows && left 2 columns
            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            if not split_block:
                horizontal_branch = masked_conv_layer2d(
                    hor_padded_x,
                    filter_size,
                    in_channel + out_channel // 2,
                    out_channel // 2,
                    'A2',
                    name + '_hor',
                    dtype=dtype,
                    padding='VALID',
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)
            else:
                horizontal_branch = split_conv(
                    hor_padded_x,
                    filter_size,
                    in_channel + out_channel // 2,
                    out_channel // 2,
                    dtype=dtype,
                    layer_name=name + '_hor',
                    activation=activation,
                    padding='VALID',
                    mask_type='A2',
                    residual=False, bn_phase=bn_phase,
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)

            out = tf.concat([vertical_branch, horizontal_branch], 3)
            if BN:
                out = batch_norm(out, phase=bn_phase, scope='bn_out')

            out = activation(out)
        elif block_type == 'mid':
            assert in_channel % 2 == 0
            assert out_channel % 2 == 0
            assert in_channel == out_channel
            vertical_branch = x[:, :, :, :in_channel // 2]
            horizontal_branch = x[:, :, :, in_channel // 2:]
            # Should add padding, top 2 rows
            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            if not split_block:
                vertical_branch = conv_layer2d(
                    ver_padded_x,
                    filter_size,
                    in_channel // 2,
                    out_channel // 2,
                    name + '_ver',
                    padding='VALID',
                    dtype=dtype,
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)
            else:
                vertical_branch = split_conv(ver_padded_x,
                                             filter_size,
                                             in_channel // 2,
                                             out_channel // 2,
                                             dtype=dtype,
                                             layer_name=name + '_ver',
                                             activation=activation,
                                             padding='VALID',
                                             mask_type=None,
                                             residual=True,
                                             bn_phase=bn_phase,
                                             x_before_pad=vertical_branch,
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)

            # If the activation below applies here, then
            # it is applied before the down_shift concatanation step
            # This might not be what we want
            #
            # vertical_branch = activation(vertical_branch)

            # Wrong CONCATE WAY
            # horizontal_branch = tf.concat([horizontal_branch[:,0:1,:,:],
            #                                horizontal_branch[:,1:,:,:] + vertical_branch[:,:-1,:,:]],
            #                               axis=1)
            # Correct Concate way
            # N H W C
            down_shift_v_branch = tf.pad(vertical_branch[:, :-1, :, :],
                                         [[0, 0], [1, 0], [0, 0], [0, 0]],
                                         "CONSTANT")
            horizontal_branch = tf.concat(
                [down_shift_v_branch, horizontal_branch], axis=3)

            # Should add padding, top 2 rows && left 2 columns
            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            if not split_block:
                horizontal_branch = conv_layer2d(
                    hor_padded_x,
                    filter_size,
                    in_channel // 2 + out_channel // 2,
                    out_channel // 2,
                    name + '_hor',
                    padding='VALID',
                    dtype=dtype,
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)
            else:
                horizontal_branch = split_conv(
                    hor_padded_x,
                    filter_size,
                    in_channel // 2 + out_channel // 2,
                    out_channel // 2,
                    dtype=dtype,
                    layer_name=name + '_hor',
                    activation=activation,
                    padding='VALID',
                    mask_type=None,
                    residual=True,
                    bn_phase=bn_phase,
                    x_before_pad=horizontal_branch,
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)

            # horizontal_branch = masked_conv_layer2d(hor_padded_x, filter_size, in_channel//2+out_channel//2, out_channel//2,
            #                                         'A2', name+'_hor', dtype=dtype, padding='VALID',
            #                                         layer_collection=layer_collection,
            #                                         registered=registered)

            if BN:
                vertical_branch = batch_norm(vertical_branch,
                                             phase=bn_phase,
                                             scope='bn_ver')
                horizontal_branch = batch_norm(horizontal_branch,
                                               phase=bn_phase,
                                               scope='bn_hor')

            vertical_branch = activation(vertical_branch)
            horizontal_branch = activation(horizontal_branch)
            if residual_connection:
                horizontal_branch = horizontal_branch + x[:, :, :,
                                                          in_channel // 2:]

            out = tf.concat([vertical_branch, horizontal_branch], 3)
        elif block_type == 'end':
            assert in_channel % 2 == 0
            assert out_channel % 2 == 0
            vertical_branch = x[:, :, :, :in_channel // 2]
            horizontal_branch = x[:, :, :, in_channel // 2:]
            # Should add padding, top 2 rows
            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            if not split_block:
                vertical_branch = conv_layer2d(
                    ver_padded_x,
                    filter_size,
                    in_channel // 2,
                    in_channel // 2,
                    name + '_ver',
                    padding='VALID',
                    dtype=dtype,
                    weight_normalization=weight_normalization,
                    layer_collection=layer_collection,
                    registered=registered)
            else:
                vertical_branch = split_conv(ver_padded_x,
                                             filter_size,
                                             in_channel // 2,
                                             in_channel // 2,
                                             dtype=dtype,
                                             layer_name=name + '_ver',
                                             activation=activation,
                                             padding='VALID',
                                             mask_type=None,
                                             residual=True,
                                             bn_phase=bn_phase,
                                             x_before_pad=vertical_branch,
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)

            # If the activation below applies here, then
            # it is applied before the down_shift concatanation step
            # This might not be what we want
            # vertical_branch = activation(vertical_branch)

            # Wrong CONCATE WAY
            # horizontal_branch = tf.concat([horizontal_branch[:,0:1,:,:],
            #                                horizontal_branch[:,1:,:,:] + vertical_branch[:,:-1,:,:]],
            #                               axis=1)
            # Correct Concate way
            # N H W C
            down_shift_v_branch = tf.pad(vertical_branch[:, :-1, :, :],
                                         [[0, 0], [1, 0], [0, 0], [0, 0]],
                                         "CONSTANT")

            horizontal_branch = tf.concat(
                [down_shift_v_branch, horizontal_branch], axis=3)

            # Should add padding, top 2 rows && left 2 columns
            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            horizontal_branch = conv_layer2d(hor_padded_x,
                                             filter_size,
                                             in_channel // 2 + in_channel // 2,
                                             out_channel,
                                             name + '_hor',
                                             padding='VALID',
                                             dtype=dtype,
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)
            # horizontal_branch = masked_conv_layer2d(hor_padded_x, filter_size, in_channel//2+out_channel//2, 4,
            #                                         'A2', name+'_hor', dtype=dtype, padding='VALID',
            #                                         layer_collection=layer_collection,
            #                                         registered=registered)

            out = horizontal_branch
        else:
            raise NotImplementedError

        return out


def pixel_resiual_block(x, block_name, dtype, filter_size, activation, num_of_layers=2,
                        layer_collection=None, registered=False, weight_normalization=False):
    x_shape = x.get_shape().as_list()
    x_input = x
    num_channel = x_shape[3]
    for idx in range(num_of_layers - 1):
        x = pixel_block_sharir_v2(x, num_channel, num_channel, 'mid', block_name+'-'+str(idx),
                                  dtype=dtype, filter_size=filter_size,
                                  activation=activation,
                                  layer_collection=layer_collection,
                                  registered=registered,
                                  weight_normalization=weight_normalization,
                                  )
        activation(x)

    x = pixel_block_sharir_v2(x, num_channel, num_channel, 'mid', block_name+'-'+str(num_of_layers-1),
                              dtype=dtype, filter_size=filter_size,
                              activation=activation,
                              layer_collection=layer_collection,
                              registered=registered,
                              weight_normalization=weight_normalization,
                              )
    x = x + x_input
    activation(x)
    return x


def pixel_block_sharir_v2(x,
                          in_channel,
                          out_channel,
                          block_type,
                          name,
                          dtype,
                          filter_size=3,
                          activation=tf.nn.relu,
                          layer_collection=None,
                          registered=False,
                          weight_normalization=False):
    '''
    for starting block, input: x, output: out with two branch concat in channel dimension
    for mid block,  input x with two branch concat in channel dimension
                    output with two branch concat in channel dimension.
    for end block,  input x with two branch concat in channel dimension
                    output with 4 channel, representing spin up spin down amp = exp(r+i\theta)
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Starting block
        if block_type == 'start':
            assert out_channel % 4 == 0
            vertical_branch = x
            horizontal_branch = x

            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            vertical_branch = conv_layer2d(ver_padded_x,
                                           filter_size,
                                           in_channel,
                                           out_channel // 2,
                                           name + '_ver',
                                           dtype=dtype,
                                           padding='VALID',
                                           weight_normalization=weight_normalization,
                                           layer_collection=layer_collection,
                                           registered=registered)
            # Leaving the vertical_branch being pre-act

            y = activation(vertical_branch)
            # N H W C
            down_shift_v_branch = tf.pad(y[:, :-1, :, :],
                                         [[0, 0], [1, 0], [0, 0], [0, 0]],
                                         "CONSTANT")
            down_shift_v_branch = conv_layer2d(down_shift_v_branch,
                                               1,
                                               out_channel // 2,
                                               out_channel // 4,
                                               name + '_ver_2_concat',
                                               dtype=dtype,
                                               padding='VALID',
                                               weight_normalization=weight_normalization,
                                               layer_collection=layer_collection,
                                               registered=registered)
            down_shift_v_branch = activation(down_shift_v_branch)

            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            horizontal_branch = masked_conv_layer2d(hor_padded_x,
                                                    filter_size,
                                                    in_channel,
                                                    out_channel // 2,
                                                    'A2',
                                                    name + '_hor',
                                                    dtype=dtype,
                                                    padding='VALID',
                                                    weight_normalization=weight_normalization,
                                                    layer_collection=layer_collection,
                                                    registered=registered)
            horizontal_branch = activation(horizontal_branch)
            horizontal_branch = conv_layer2d(horizontal_branch,
                                             1,
                                             out_channel // 2,
                                             out_channel // 4,
                                             name + '_hor_2_concat',
                                             dtype=dtype,
                                             padding='VALID',
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)
            horizontal_branch = activation(horizontal_branch)

            horizontal_branch = tf.concat(
                [down_shift_v_branch, horizontal_branch], axis=3)

            # Should add padding, top 2 rows && left 2 columns
            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            horizontal_branch = conv_layer2d(hor_padded_x,
                                             filter_size,
                                             out_channel // 2,
                                             out_channel // 2,
                                             name + '_hor2',
                                             dtype=dtype,
                                             padding='VALID',
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)

            out = tf.concat([vertical_branch, horizontal_branch], 3)
        elif block_type == 'mid':
            assert in_channel % 2 == 0
            assert out_channel % 2 == 0
            assert in_channel == out_channel
            vertical_branch = x[:, :, :, :in_channel // 2]
            horizontal_branch = x[:, :, :, in_channel // 2:]
            # Should add padding, top 2 rows

            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            vertical_branch = conv_layer2d(ver_padded_x,
                                           filter_size,
                                           in_channel // 2,
                                           out_channel // 2,
                                           name + '_ver',
                                           dtype=dtype,
                                           padding='VALID',
                                           weight_normalization=weight_normalization,
                                           layer_collection=layer_collection,
                                           registered=registered)
            # Leaving the vertical_branch being pre-act

            y = activation(vertical_branch)
            # N H W C
            down_shift_v_branch = tf.pad(y[:, :-1, :, :],
                                         [[0, 0], [1, 0], [0, 0], [0, 0]],
                                         "CONSTANT")
            down_shift_v_branch = conv_layer2d(down_shift_v_branch,
                                               1,
                                               out_channel // 2,
                                               out_channel // 4,
                                               name + '_ver_2_concat',
                                               dtype=dtype,
                                               padding='VALID',
                                               weight_normalization=weight_normalization,
                                               layer_collection=layer_collection,
                                               registered=registered)
            down_shift_v_branch = activation(down_shift_v_branch)

            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            horizontal_branch = conv_layer2d(hor_padded_x,
                                             filter_size,
                                             in_channel // 2,
                                             out_channel // 2,
                                             name + '_hor',
                                             dtype=dtype,
                                             padding='VALID',
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)

            horizontal_branch = activation(horizontal_branch)
            horizontal_branch = conv_layer2d(horizontal_branch,
                                             1,
                                             out_channel // 2,
                                             out_channel // 4,
                                             name + '_hor_2_concat',
                                             dtype=dtype,
                                             padding='VALID',
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)
            horizontal_branch = activation(horizontal_branch)

            horizontal_branch = tf.concat(
                [down_shift_v_branch, horizontal_branch], axis=3)

            # Should add padding, top 2 rows && left 2 columns
            hor_padded_x = tf.pad(
                horizontal_branch,
                [[0, 0], [filter_size - 1, 0], [filter_size - 1, 0], [0, 0]],
                "CONSTANT")
            horizontal_branch = conv_layer2d(hor_padded_x,
                                             filter_size,
                                             out_channel // 2,
                                             out_channel // 2,
                                             name + '_hor2',
                                             dtype=dtype,
                                             padding='VALID',
                                             weight_normalization=weight_normalization,
                                             layer_collection=layer_collection,
                                             registered=registered)

            out = tf.concat([vertical_branch, horizontal_branch], 3)
        else:
            raise NotImplementedError

        return out


def pixel_block(x,
                in_channel,
                out_channel,
                block_type,
                name,
                dtype,
                filter_size=3,
                activation=tf.nn.relu,
                layer_collection=None,
                registered=False):
    '''
    for starting block, input: x, output: out with two branch concat in channel dimension
    for mid block,  input x with two branch concat in channel dimension
                    output with two branch concat in channel dimension.
    for end block,  input x with two branch concat in channel dimension
                    output with 4 channel, representing spin up spin down amp = exp(r+i\theta)
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Starting block
        if block_type == 'start':
            assert out_channel % 2 == 0
            # Should add padding, top 2 rows
            ver_padded_x = tf.pad(
                x, [[0, 0], [filter_size - 1, 0],
                    [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            vertical_branch = conv_layer2d(ver_padded_x,
                                           filter_size,
                                           in_channel,
                                           out_channel // 2,
                                           name + '_ver',
                                           dtype=dtype,
                                           padding='VALID',
                                           layer_collection=layer_collection,
                                           registered=registered)
            vertical_branch = activation(vertical_branch)
            # Should add shift and then
            # Should add padding, top 2 rows && left 2 columns
            # hor_padded_x = tf.pad(x, [[0, 0], [2, 0], [2, 0], [0, 0]], "CONSTANT")

            # This is an alternative implementation
            horizontal_branch = masked_conv_layer2d(
                x,
                filter_size,
                in_channel,
                out_channel // 2,
                'A',
                name + '_hor',
                dtype=dtype,
                layer_collection=layer_collection,
                registered=registered)
            horizontal_branch = activation(horizontal_branch)
            out = tf.concat([vertical_branch, horizontal_branch], 3)

        elif block_type == 'mid':
            assert in_channel % 2 == 0
            assert out_channel % 2 == 0
            vertical_branch = x[:, :, :, :in_channel // 2]
            horizontal_branch = x[:, :, :, in_channel // 2:]
            # Should add padding, top 2 rows
            ver_padded_x = tf.pad(
                vertical_branch,
                [[0, 0], [filter_size - 1, 0],
                 [filter_size // 2, filter_size // 2], [0, 0]], "CONSTANT")
            vertical_branch = conv_layer2d(ver_padded_x,
                                           filter_size,
                                           in_channel // 2,
                                           out_channel // 2,
                                           name + '_ver',
                                           padding='VALID',
                                           dtype=dtype,
                                           layer_collection=layer_collection,
                                           registered=registered)
            vertical_branch = activation(vertical_branch)
            # N H W C
            horizontal_branch = tf.concat([
                horizontal_branch[:, 0:1, :, :],
                horizontal_branch[:, 1:, :, :] + vertical_branch[:, :-1, :, :]
            ],
                axis=1)
            # Should add padding, top 2 rows && left 2 columns
            horizontal_branch = masked_conv_layer2d(
                horizontal_branch,
                filter_size,
                in_channel // 2,
                out_channel // 2,
                'B',
                name + '_hor',
                dtype=dtype,
                layer_collection=layer_collection,
                registered=registered)
            horizontal_branch = activation(horizontal_branch)
            out = tf.concat([vertical_branch, horizontal_branch], 3)
        elif block_type == 'end':
            assert in_channel % 2 == 0
            assert out_channel == 4
            horizontal_branch = x[:, :, :, in_channel // 2:]
            horizontal_branch = masked_conv_layer2d(
                horizontal_branch,
                filter_size,
                in_channel // 2,
                out_channel,
                'B',
                name + '_hor',
                dtype=dtype,
                layer_collection=layer_collection,
                registered=registered)
            out = horizontal_branch
        else:
            raise NotImplementedError

        return out


def split_conv(x,
               filter_size,
               in_channel,
               out_channel,
               dtype,
               layer_name,
               activation=tf.nn.relu,
               padding='VALID',
               mask_type=None,
               residual=True,
               bn_phase=None,
               x_before_pad=None,
               weight_normalization=False,
               layer_collection=None,
               registered=None):
    '''
    split with bottleneck channel = 4
    128 --> (4*8) --> 64
    64 --> (4*4) --> 64
    32+2 --> (4*4) --> 32
    '''
    assert padding == 'VALID'
    cardinality = in_channel // 8
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        layers_split = []
        for idx in range(cardinality):
            splits = bottleneck(x,
                                filter_size=filter_size,
                                in_channel=in_channel,
                                out_channel=out_channel,
                                name='split_' + str(idx),
                                activation=activation,
                                bottleneck_channel=4,
                                dtype=dtype,
                                mask_type=mask_type,
                                bn_phase=bn_phase,
                                weight_normalization=weight_normalization,
                                layer_collection=layer_collection,
                                registered=registered)
            layers_split.append(splits)

        out = tf.math.add_n(layers_split)
        if residual == True:
            assert x_before_pad is not None
            if in_channel == out_channel:
                out = out + x_before_pad
            else:
                shortcut = conv_layer2d(x_before_pad, 1, in_channel,
                                        out_channel,
                                        'split_shortcut',
                                        padding=padding,
                                        dtype=dtype,
                                        weight_normalization=weight_normalization,
                                        layer_collection=layer_collection,
                                        registered=registered)
                out = out + shortcut

        return out


def bottleneck(x,
               filter_size,
               in_channel,
               out_channel,
               dtype,
               name,
               activation=tf.nn.relu,
               padding='VALID',
               bottleneck_channel=None,
               mask_type=None,
               bn_phase=None,
               weight_normalization=False,
               layer_collection=None,
               registered=None):
    '''
    Default: the bottleneck channel would be input channel devide by 4.
    In certain situation the bottleneck channel might be given as a
    fixed number. This is used in the split_block structure in ResNeXt.

    The number output channels is not necessary equal to in the number
    of output channels.

    E.G.
    128 --> 32 --> 64
    64 --> 16 --> 64
    '''
    assert padding == 'VALID'
    if bottleneck_channel is None:
        bottleneck_channel = in_channel // 4

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = conv_layer2d(x,
                         1,
                         in_channel,
                         bottleneck_channel,
                         name + '_conv1',
                         padding=padding,
                         dtype=dtype,
                         weight_normalization=weight_normalization,
                         layer_collection=layer_collection,
                         registered=registered)
        x = batch_norm(x, phase=bn_phase, scope=name+'_bn1')
        x = activation(x)
        if mask_type is None:
            x = conv_layer2d(x,
                             filter_size,
                             bottleneck_channel,
                             bottleneck_channel,
                             name + '_conv2',
                             padding=padding,
                             dtype=dtype,
                             weight_normalization=weight_normalization,
                             layer_collection=layer_collection,
                             registered=registered)
        else:
            x = masked_conv_layer2d(x,
                                    filter_size,
                                    bottleneck_channel,
                                    bottleneck_channel,
                                    mask_type,
                                    name + '_mask_conv2',
                                    dtype=dtype,
                                    padding='VALID',
                                    weight_normalization=weight_normalization,
                                    layer_collection=layer_collection,
                                    registered=registered)

        x = batch_norm(x, phase=bn_phase, scope=name+'_bn2')
        x = activation(x)
        x = conv_layer2d(x,
                         1,
                         bottleneck_channel,
                         out_channel,
                         name + '_conv3',
                         padding=padding,
                         dtype=dtype,
                         weight_normalization=weight_normalization,
                         layer_collection=layer_collection,
                         registered=registered)

        x = batch_norm(x, phase=bn_phase, scope=name+'_bn3')
    return x


def bottleneck_residual(x, in_channel, out_channel, name, stride_size=2):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # Identity shortcut
        if in_channel == out_channel:
            shortcut = x
            x = self.conv_layer2d(x, 1, in_channel, out_channel / 4, "conv1")
            # conv projection shortcut
        else:
            shortcut = x
            shortcut = self.conv_layer2d(shortcut,
                                         1,
                                         in_channel,
                                         out_channel,
                                         "shortcut",
                                         stride_size=stride_size)
            shortcut = self.batch_norm(shortcut,
                                       phase=self.bn_is_training,
                                       scope='shortcut/bn')
            x = self.conv_layer2d(x,
                                  1,
                                  in_channel,
                                  out_channel / 4,
                                  "conv1",
                                  stride_size=stride_size)

        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn1')
        x = tf.nn.relu(x)
        x = self.conv_layer2d(x, 3, out_channel / 4, out_channel / 4, "conv2")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn2')
        x = tf.nn.relu(x)
        x = self.conv_layer2d(x, 1, out_channel / 4, out_channel, "conv3")
        x = self.batch_norm(x, phase=self.bn_is_training, scope='bn3')
        x += shortcut
        x = tf.nn.relu(x)

    return x


def residual_block(x,
                   num_channel,
                   name,
                   stride_size=1,
                   activation=tf.nn.relu,
                   bn_is_training=True):
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
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # conv projection shortcut
        shortcut = x
        x = circular_conv_2d(x,
                             3,
                             in_channel,
                             out_channel,
                             "conv1",
                             stride_size=stride_size,
                             biases=True)
        x = batch_norm(x, phase=bn_is_training, scope='bn1')
        x = activation(x)
        x = circular_conv_2d(x,
                             3,
                             in_channel,
                             out_channel,
                             "conv2",
                             stride_size=stride_size,
                             biases=True)
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
    initial_value = tf.random_uniform(j_factor_size,
                                      minval=-.1 * scale,
                                      maxval=.1 * scale,
                                      dtype=dtype)
    weights = get_var(initial_value, name + "weights", dtype=dtype)
    weights_upper = tf.matrix_band_part(weights, 0, -1)
    weights_symm = 0.5 * (weights_upper + tf.transpose(weights_upper))
    return weights_symm


def jastrow_2d_amp(config_array, Lx, Ly, local_d, name, sym=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        total_dim = Lx * Ly * local_d
        # get symmetry weights matrix, (total_dim x total_dim )
        weights_symm_re = get_jastrow_var(2,
                                          total_dim,
                                          name="real_",
                                          dtype=tf.float32)
        weights_symm_im = get_jastrow_var(2,
                                          total_dim,
                                          name="imag_",
                                          dtype=tf.float32,
                                          scale=15)

        config_vector = tf.reshape(config_array, [-1, total_dim])
        C2_array = tf.einsum('ij,ik->ijk', config_vector, config_vector)
        C2_array = tf.multiply(tf.complex(C2_array, tf.zeros_like(C2_array)),
                               tf.complex(weights_symm_re, weights_symm_im))
        amp_array = tf.reduce_sum(C2_array, axis=[1, 2], keep_dims=False)

    return tf.exp(amp_array)


def jacobian(y, x):
    y_flat = tf.reshape(y, (-1, ))
    jacobian_flat = tf.stack(
        [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
    return tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))
