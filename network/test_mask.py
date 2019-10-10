import tf_wrapper as tf_
import mask
import tf_network
import tensorflow as tf

sess = tf.InteractiveSession()

Lx = Ly = 4
alpha = 2
LxLy = Lx * Ly
ordering = mask.gen_raster_scan_order(Lx)

x = tf.constant(np.random.rand(Lx, Ly, 2))
x_reshaped = tf.reshape(x, [-1, LxLy, 2])
x = tf.cast(x, dtype=tf.float64)
fc1 = tf_.masked_fc_layer(x, LxLy * 2, LxLy * alpha, 'masked_fc1',
                      ordering, 'A', dtype=tf.float64)
fc1 = act(fc1)
fc2 = tf_.masked_fc_layer(fc1, self.LxLy * self.alpha, self.LxLy * self.alpha,
                      'masked_fc2', self.ordering, 'B',
                      layer_collection=self.layer_collection,
                      registered=self.registered, dtype=self.TF_FLOAT)
fc2 = act(fc2)
