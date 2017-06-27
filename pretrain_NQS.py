import tensorflow as tf
import numpy as np
from tf_NN import tf_NN
from tf_NN3 import tf_NN3
from tf_CNN import tf_CNN
from tf_FCN import tf_FCN

L = 10
which_Net = 'NN3'
basis = []
for line in open('EigenVec/basisMatrix'+str(L)+'.csv', 'r'):
    basis.append(line[:-1])

newbasis = np.zeros((len(basis), L, 2))
for i in range(len(basis)):
    for j in range(L):
        newbasis[i, j, 0] = basis[i][j]

amp_array = np.genfromtxt(open('EigenVec/EigVec_L'+str(L)+'V20W0E0.csv', 'r'))

systemSize = (L, 2)
if which_Net == 'NN':
    Net = tf_NN(systemSize, optimizer='Mom', alpha=4)
elif which_Net == 'NN3':
    Net = tf_NN3(systemSize, optimizer='Mom', alpha=2)
elif which_Net == 'CNN':
    Net = tf_CNN(systemSize, optimizer='Mom')
elif which_Net == 'FCN':
    Net = tf_FCN(systemSize, optimizer='Mom')

num_train = 2**L
to_large_dict = {}
X = np.zeros((num_train, L, 2))
for i in range(0, num_train):
    temp = i+1
    for j in range(L):
        if temp > 2**(L-j-1):
            X[i, j, 0] = 1
            temp -= 2**(L-j-1)
        else:
            X[i, j, 1] = 1
            pass
    to_large_dict[''.join([str(int(ele)) for ele in X[i, :, 0]])] = i

X = X.reshape((num_train, L*2))
Y = np.zeros((num_train, 1))
for i in range(len(amp_array)):
    idx = to_large_dict[''.join([str(int(ele)) for ele in newbasis[i, :, 0]])]
    Y[idx] = amp_array[i]

# X_nz = X[(np.abs(Y) * np.ones((1, 32)) >= 1e-15)]
# Y_nz = Y[(np.abs(Y) >= 1e-15)]
# X = X_nz.reshape((len(Y_nz), 32))
# Y = Y_nz.reshape((len(Y_nz),1))
Y = Y
print X.shape, Y.shape

with Net.sess as sess:

    true_out = tf.placeholder(tf.float32, [None, 1])

    # xent = tf.nn.softmax_cross_entropy_with_logits(logits=Net.pred, labels=true_out)
    # cost = tf.reduce_mean(xent, name='xent')
    cost = tf.reduce_mean((Net.pred - true_out)*(Net.pred - true_out))
    lr = 0.01
    learning_rate = tf.Variable(lr)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
    sess.run(tf.global_variables_initializer())

    print len(Net.model_var_list), len(Net.para_list)
    saver = tf.train.Saver()  # Net.model_var_list)
    ckpt = tf.train.get_checkpoint_state('Model/'+which_Net+'/L'+str(L))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(Net.sess, ckpt.model_checkpoint_path)
        print("Restore from last check point")
    else:
        print("No checkpoint found")

    batch_size = 128
    min_c = 10000
    #    sess.run(learning_rate.assign(0.1 * (0.8**(i/10000))))
    for i in xrange(100000):
        batch_mask = np.random.choice(len(Y), batch_size)  # ,p=Y*Y)
        # _, c = sess.run([train_step, cost], feed_dict={Net.x: X[batch_mask],
        #                                                true_out: Y[batch_mask]})
        _, c = sess.run([train_step, cost], feed_dict={Net.x: X, true_out: Y})
        if i % 500 == 0:
            print "step:", i, " L2 error:", c
#            if min_c > c:
#                min_c = c
            saver.save(sess, 'Model/'+which_Net+'/L'+str(L)+'/pre')
        if i % 5000 == 0:
            y = sess.run(Net.pred, feed_dict={Net.x: X})
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(Y, '-o')
            plt.plot(y, '--')
            plt.show()

    # fig.savefig('L16_pretrain.eps',bbox_inches='tight')
