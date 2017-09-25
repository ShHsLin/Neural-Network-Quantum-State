import tensorflow as tf
import numpy as np

Lx = 6
in_channel = 2
out_channel = 3
filter_size = Lx
# X = [batch_size, Lx, in_channel]
np_X = np.random.rand(1, Lx, in_channel)
tf_X = tf.Variable(np_X, dtype=tf.float32)

# W = [filter_size, in_channel, out_channel]
np_W = np.random.rand(filter_size, in_channel, out_channel)
tf_W = tf.Variable(np_W, dtype=tf.float32)


np_a = np.random.rand(10)
tf_a = tf.Variable(np_a, dtype=tf.float32)
# bias = np.array([3.9, 4.2, -4, -3, -5, 5, 10, 12, 15, -3])
bias = np.array([3.9, 4.2, -4, 0, 0, 0, 0, 0., 0., 0])
tf_bias = tf.Variable(bias, dtype=tf.float32)
tf_eye = tf.Variable(np.eye(10))
eye = np.eye(10)
per_mat = [eye]
for i in xrange(1, 10):
    per_mat.append(np.concatenate([eye[:, 10-i:], eye[:, :10-i]], 1))

per_mat = np.stack(per_mat, 2)
# print per_mat, per_mat.shape

per_a = np_a.dot(per_mat)
print np_a.dot(per_mat), np_a.dot(per_mat).shape

print bias.dot(per_a)
print np.fft.ifft(np.einsum('i,i->i', np.conj(np.fft.fft(bias)), np.fft.fft(np_a)))
print np.fft.ifft(np.einsum('i,i->i', np.fft.fft(bias), np.conj(np.fft.fft(np_a))))

a_fft = tf.fft(tf.complex(tf_a, 0.))
b_fft = tf.fft(tf.complex(tf_bias, 0.))
conv = tf.ifft(tf.conj(a_fft) * b_fft)


tf_X_pad = tf.concat([tf_X, tf_X[:, :filter_size-1, :]], 1)
circular_conv = tf.nn.conv1d(tf_X_pad, tf_W, 1, padding='VALID')

# i:batch_size, j:Lx, k:in_channel, l:out_channel #
tf_X_fft = tf.fft(tf.complex(tf.einsum('ijk->ikj', tf_X), 0.))
tf_W_fft = tf.fft(tf.complex(tf.einsum('jkl->klj', tf_W), 0.))
tf_XW_fft = tf.einsum('ikj,klj->iklj', (tf_X_fft), tf.conj(tf_W_fft))
tf_XW = tf.einsum('iklj->ijl', tf.ifft(tf_XW_fft))


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # print sess.run(conv)
    print sess.run(circular_conv) - sess.run(tf.real(tf_XW))
