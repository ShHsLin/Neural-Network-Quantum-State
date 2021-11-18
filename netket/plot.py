import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import seaborn as sns


x_collections = []
y_collections = []
for filename in sys.argv[1:]:
    data = pickle.load(open(filename, 'rb'))
    np_timesteps = data['timesteps']
    np_overlaps = data['overlaps']
    x_collections.append(np_timesteps)
    y_collections.append(1. - np_overlaps)

#     plt.semilogy(np_timesteps, 1. - np_overlaps)
# 
# plt.ylabel("Overlap")
# # plt.xlabel("Iteration #")
# plt.xlabel("Run time")
# plt.axhline(y=1e-4, xmin=0, xmax=len(np_overlaps), linewidth=2, color="k", label="1")
# plt.title(r"Transverse-field Ising model") # , $L=" + str(L) + "$")
# plt.show()


combine_timesteps = np.stack(x_collections, axis=-1)
combine_np_overlaps = np.stack(y_collections, axis=-1)
x_mean = np.mean(combine_timesteps, axis=-1)
y_mean = np.mean(combine_np_overlaps, axis=-1)
y_std = np.std(combine_np_overlaps, axis=-1)

# df = pd.DataFrame({'x': x_mean, 'y': y_mean, 'y_std': 2*y_std})
fig, ax = plt.subplots(figsize=(9,5))
# sns.lineplot(df, x='x', y='y', ci='y_std', err_style='band')
# plt.errorbar(x_mean, y_mean, y_std)
plt.plot(x_mean, y_mean)
plt.fill_between(x_mean, y_mean-y_std, y_mean+y_std, facecolor='r',alpha=0.5)

plt.yscale('log')
plt.show()

