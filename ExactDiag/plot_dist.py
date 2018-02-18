import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
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
# sns.set(font="DejaVu Sans")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

def plot_fig_4(filename, inset_scale=0.05, restricted=False):
    amp=np.genfromtxt(filename)
    log_abs_amp=np.log(np.abs(amp))

    if restricted:
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


    fig, axes = plt.subplots(1, 2, sharex=False)
    sns.distplot(amp, color="r", ax=axes[0])
    axes[0].set_title(r"$(a)$ Distribution of $C_i$", y=1.02)
    axes[0].set_xlabel(r"$C_i$")
    axes[0].set_ylabel(r"$P(C_i)$")
    sns.distplot(log_abs_amp, color="b", ax=axes[1])
    axes[1].set_title(r"$(b)$ Distribution of $\log|C_i|$", y=1.02)
    axes[1].set_xlabel(r"$\log|C_i|$")
    axes[1].set_ylabel(r"$P(\log|C_i|)$")

    axes0_pos = axes[0].get_position()
    # left, bottom, width, height #
    inset_pos = [axes0_pos.x0 + axes0_pos.width - 0.21,
                 axes0_pos.y0 + axes0_pos.height - 0.225,
                 0.2,
                 0.2]
    inset_ax = fig.add_axes(inset_pos)
    inset_ax.xaxis.set_tick_params(labelsize=24)
    inset_ax.yaxis.set_tick_params(labelsize=24)
    sns.distplot(amp, color="r", ax=inset_ax)
    inset_ax.set_ylim([0,inset_scale])

    # plt.setp(axes, yticks=[])
    # plt.tight_layout()
    eps_name = filename[10:-4]
    if restricted:
       final_name = 'dist_' + 'sz0_' + eps_name + '.pdf'
    else:
       final_name = 'dist_' + eps_name + '.pdf'

    plt.savefig(final_name)
    # plt.show()

plot_fig_4('EigVec/ES_2d_L4x4_J2_0.csv', inset_scale=0.02)
plot_fig_4('EigVec/ES_2d_L4x4_J2_5.csv', inset_scale=0.03)
plot_fig_4('EigVec/ES_2d_L4x4_J2_0.csv', inset_scale=0.08, restricted=True)
plot_fig_4('EigVec/ES_2d_L4x4_J2_5.csv', inset_scale=0.08, restricted=True)

