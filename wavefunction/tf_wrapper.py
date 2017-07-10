import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, padding='SAME'):
    # Conv2D wrapper, with bias
    # Without relu operation
    # y strides = 2 to contain only valid convolution
    x = tf.nn.conv2d(x, W, strides=[1, strides, 2, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return x


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


def maxpool1d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                          padding='VALID')


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


def avgpool1d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1],
                          padding='VALID')


def avgpool2d(x, k=2):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')
