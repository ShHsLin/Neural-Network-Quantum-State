import tensorflow as tf
import numpy as np

np_a = np.arange(10)
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


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(conv)
