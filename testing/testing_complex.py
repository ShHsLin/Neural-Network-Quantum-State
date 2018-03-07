import tensorflow as tf
import numpy as np

z = tf.placeholder(tf.complex64)
sess = tf.InteractiveSession()
print("z = 2+0.1j")
print("log(z)")
print(sess.run(tf.log(z), feed_dict={z:2+0.1j}))

print("tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))
print("tf.gradients(tf.conj(z)*z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.conj(z)*z, z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))
# print(sess.run(tf.gradients(tf.exp(z), z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.complex(x,y)
print("tf.gradients(z*z, [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*z, [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.real(z*z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.real(z*z), [x,y]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.imag(z*z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.imag(z*z), [x,y]),
               feed_dict={x:2,y:0.1}))




'''
print("Testing complex gradient descent")
x = -2+1j
for i in range(100):
    print('now value: ',sess.run(tf.real(tf.exp(-tf.conj(z)*z)), feed_dict={z:x}))
    gradient = sess.run(tf.gradients(tf.real(tf.exp(-tf.conj(z)*z)), z), feed_dict={z: x})
    print('x : ', x, 'gradient : ', gradient[0])
    x = x + ( gradient[0] * 0.5)
'''
