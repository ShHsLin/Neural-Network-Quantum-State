# Neural Network Quantum State (NNQS)

This repository contains sourse code used in [Scaling of neural-network quantum states for time evolution](https://arxiv.org/abs/2104.10696)

The project aim to treat neural network as an tensor approximation tool by training the neural network with one regression output, i.e. the tensor value.

The original motivation is to study the obtain the ground state of quantum many body problems, where the ground state is a high dimensional tensor.


### Download
```
git clone --recurse-submodules git@github.com:ShHsLin/Neural-Network-Quantum-State.git
```

### Requirements

  - python 2.7 or >= 3
  - numpy, tensorflow 1.14
  Note that with tensorflow <= 1.3, Jastrow wavefunction seems not to work properly.
  Note that the code is written with tensorflow 1 and does not support tensorflow 2.


### How to run the Code

To run the supervised learning, one first need to provide an numpy array containing the target probability amplitudes. For example, one can generate time-evolved states following global quenches with exact diagonalization.
```
cd ExactDiag
python many_body.py 1dZZ_X_XX_TE 16 3 1
```
The wavefunction is saved under the directory wavefunction. Then going back to the original directory, and run
```
python supervised_NQS.py --l 20 --dim 1 --net ${net} --using_complex 1 --Q_tar -1  --lr 1e-3 --opt Adam --path ${NN_PATH} --act relu --alpha ${alpha}  --T ${T}  --supervised_model ${supervised_model} --filter_size ${fsize} --num_blocks ${blocks}
```

net=gated_CNN
alpha=3
T=0.04
fsize=3
num_block=10
NN_path is the path to the directory for saving data.



     

### Citation

To cite our work, please copy
```
@article{lin2021scaling,
  title={Scaling of neural-network quantum states for time evolution},
  author={Lin, Sheng-Hsuan and Pollmann, Frank},
  journal={arXiv preprint arXiv:2104.10696},
  year={2021}
}
```
