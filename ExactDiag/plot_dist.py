import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import ticker ## to control the number of tick in plot
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

x=0
params = {'legend.fontsize':34-x,#'xx-large', 
                    'figure.figsize': (15,7),
                    'axes.labelsize': 38-x,
                    'axes.titlesize': 38-x,
                    'xtick.labelsize': 34-x,#'xx-large',
                    'ytick.labelsize': 34-x,#'xx-large',
                    'figure.autolayout': True,
                    'mathtext.fontset': u'cm',
                    'font.family': u'serif',
                    'font.serif': u'Times New Roman',
                    #          'pgf.texsystem':'pdflatex',
                    #          'text.usetex': True,
                    #          'text.latex.unicode': False,
                    #          'text.dvipnghack' : True
                   }

pylab.rcParams.update(params)


amp=np.genfromtxt('EigVec/ES_2d_L4x4_J2_5.csv')
log_abs_amp=np.log(np.abs(amp))

restricted_amp=[]
other_amp=[]
for large_idx in range(2**16):
    str_idx = "{0:0>16}".format("{0:b}".format(large_idx))
    sum_sz = np.sum([int(i) for i in str_idx])
    if sum_sz == 8:
        restricted_amp.append(amp[large_idx])
    else:
        other_amp.append(amp[large_idx])

amp=np.array(restricted_amp)
log_abs_amp=np.log(np.abs(amp))



# sns.set(style="white", palette="muted", color_codes=True)
# f, axes = plt.subplots(1, 2, figsize=(7, 7), sharex=False)

# sns.despine(left=True)
# sns.distplot(d, kde=False, color="b", ax=axes[0])
# sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])
# sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1])

fig, axes = plt.subplots(1, 2, sharex=False)
sns.distplot(amp, color="r", ax=axes[0])
axes[0].set_title("$(a)\ \  C_i$", y=1.02)
sns.distplot(log_abs_amp, color="b", ax=axes[1])
axes[1].set_title("$(b)\ \  \log(|C_i|)$", y=1.02)

axes0_pos = axes[0].get_position()
# left, bottom, width, height #
inset_pos = [axes0_pos.x0 + axes0_pos.width - 0.2,
             axes0_pos.y0 + axes0_pos.height - 0.225,
             0.2,
             0.2]
inset_ax = fig.add_axes(inset_pos)
inset_ax.xaxis.set_tick_params(labelsize=24)
inset_ax.yaxis.set_tick_params(labelsize=24)
sns.distplot(amp, color="r", ax=inset_ax)
inset_ax.set_ylim([0,0.08])


# plt.setp(axes, yticks=[])
# plt.tight_layout()

# plt.show()
plt.savefig('dist_sz0_2d_L4x4_J2_5.pdf')

