import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import spline


def plot_smooth_line(y_array, label=None):
    y_array = y_array[::10]
    xnew = np.linspace(0, len(y_array),1000)
    # 300 represents number of points to make between T.min
    power_smooth = spline(np.arange(len(y_array)), y_array, xnew)
    plt.plot(xnew,power_smooth, label=label)

# 'L16_NN_complex_a2_Mom1e-03_batch.csv'
# 'L16_RBM_a2_Mom1e-03_batch.csv'
# 'L16_NN_complex_a2_Mom1e-03_total.csv'
# 'L16_RBM_a2_Mom1e-03_total.csv'
net = ['NN_complex_a2', 'RBM_a2', 'NN_a2', 'NN3_complex_a1', 'NN3_a1', 'ResNet_a1']
opt = ['Mom1e-03', 'Mom1e-02']

fig = plt.figure()
for n in net:
    for o in opt:
        filename = 'L16_%s_%s_total.csv' % (n, o)
        try:
            hand = open('../log/pretrain/'+filename, "r")
        except:
            continue
        accuracy = np.genfromtxt(hand)

        plt.plot(accuracy, label=n+' '+o)

plt.legend(shadow=True, loc=0, frameon=False)
plt.ylabel('error')
plt.show()
fig.savefig('pretrain'+'.png')
