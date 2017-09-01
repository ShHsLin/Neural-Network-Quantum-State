import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fileList=['L10NNJ1','L10CNNJ1']
fileList1=['HeisenbergNNL10.txt', 'HeisenbergCNNL10.txt']
fileList2=['IsingNNL32.txt','IsingCNNL32.txt']

fig=plt.figure()
for filename in fileList:
    Energy=[]
    hand = open(filename,"r")
    for line in hand:
        number = re.findall('[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?',line)
        #number = re.findall('[-+]?([0-9]*\.[0-9]+|[0-9]+)',line)
        Energy.append(float(number[0]))

    plt.plot(Energy,label=filename)


plt.legend(shadow=True, loc=0, frameon=False)
fig.savefig('Heisenberg'+'.eps')

