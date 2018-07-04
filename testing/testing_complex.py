import tensorflow as tf
import numpy as np


sess = tf.InteractiveSession()
z = tf.placeholder(tf.float64)
print("(float64) z = -2")
print("log(z) : ", sess.run(tf.log(z), feed_dict={z:-2}))

z = tf.placeholder(tf.complex64)
print("(complex64) z = -2")
print("log(z) : ", sess.run(tf.log(z), feed_dict={z:-2}))
print("z = 2+0.1j")
print("log(z) : ", sess.run(tf.log(z), feed_dict={z:2+0.1j}))

print("tf.gradients(tf.conj(z)*z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.conj(z)*z, z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))
# print(sess.run(tf.gradients(tf.exp(z), z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))
print("tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)]), feed_dict={z:2+0.1j}))
print("tf.gradients(tf.conj(z)*tf.conj(z), z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.conj(z)*tf.conj(z), z, grad_ys=[tf.complex(1.,0.)]),
               feed_dict={z:2+0.1j}))


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = tf.complex(x,y)
zbar = tf.complex(x,-y)
print("\n\nx,y --> z = x+iy --> z*z=x^2 - y^2 + i 2xy")
print("tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*z, z, grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(z*z, [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*z, [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.real(z*z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.real(z*z), [x,y]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.imag(z*z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.imag(z*z), [x,y]),
               feed_dict={x:2,y:0.1}))


print("\n\nx,y --> z = x+iy --> z*conj(z)=x^2 + y^2")
print("tf.gradients(z*tf.conj(z), z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*tf.conj(z), z, grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(z*tf.conj(z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*tf.conj(z), [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.real(z*tf.conj(z)), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.real(z*tf.conj(z)), [x,y]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.imag(z*tf.conj(z)), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.imag(z*tf.conj(z)), [x,y]),
               feed_dict={x:2,y:0.1}))


print(" \nThis example shows that the assumption of holomorphic function in tf."
      " would lead to wrong derivative for non-holomorphic function."
      " Taking the complex conjugate of the derivative w.r.t. z "
      " is not equivalent to taking derivative w.r.t. zbar")
print("\n\nat {x:2,y:0.1}")
print("x,y --> z = x+iy --> z*conj(z) + z = x^2 + y^2 + x + iy")
print("tf.gradients(z*tf.conj(z) + z, z, grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*tf.conj(z) + z, z, grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("We can not do this:\n tf.gradients(z*tf.conj(z) + z, tf.conj(z), grad_ys=[tf.complex(1.,0.)])")
print("also we can not do this: \n tf.gradients(z*tf.conj(z) + z, zbar, grad_ys=[tf.complex(1.,0.)])")
print("tf.gradients(z*tf.conj(z) + z , [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(z*tf.conj(z) + z, [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.real(z*tf.conj(z) + z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.real(z*tf.conj(z) + z), [x,y]),
               feed_dict={x:2,y:0.1}))
print("tf.gradients(tf.imag(z*tf.conj(z) + z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.imag(z*tf.conj(z) + z), [x,y]),
               feed_dict={x:2,y:0.1}))


import pdb;pdb.set_trace()
z = tf.placeholder(tf.complex64)
print(sess.run(tf.gradients(z*tf.conj(z) + z, z, grad_ys=[tf.complex(1.,0.)]),
               feed_dict={z:2 + 0.1j}))
print(sess.run(tf.gradients(z*tf.conj(z) + z, tf.conj(z), grad_ys=[tf.complex(1.,0.)]),
               feed_dict={z:2 + 0.1j}))
import pdb;pdb.set_trace()

print("\n\nx,y --> z = x+iy --> log(z)")
print("tf.gradients(tf.log(z), [x,y], grad_ys=[tf.complex(1.,0.)])")
print(sess.run(tf.gradients(tf.log(z), [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:2,y:0.1}))
print("numpy : df/dx ", 1/(2+0.1j))
print("numpy : df/dy ", 1j/(2+0.1j))

print("x,y --> x*y --> log(x*y)")
print("tf.gradients(tf.log(x*y), [x,y]), x=-1,y=1")
print(sess.run(tf.gradients(tf.log(x*y), [x,y]),
               feed_dict={x:-1.,y:1.}))
print(sess.run(tf.gradients(tf.log(tf.cast(x*y,tf.complex64)), [x,y], grad_ys=[tf.complex(1.,0.)]),
               feed_dict={x:-1.,y:1.}))
print("exact : df/dx = 1/x ", -1.)
print("exact : df/dy = 1/y", 1.)


'''
Lession 1:
    log(negative float) --> nan
Lession 2:

'''



'''
print("Testing complex gradient descent")
x = -2+1j
for i in range(100):
    print('now value: ',sess.run(tf.real(tf.exp(-tf.conj(z)*z)), feed_dict={z:x}))
    gradient = sess.run(tf.gradients(tf.real(tf.exp(-tf.conj(z)*z)), z), feed_dict={z: x})
    print('x : ', x, 'gradient : ', gradient[0])
    x = x + ( gradient[0] * 0.5)
'''
