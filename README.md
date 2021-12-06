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
python many_body.py 1dZZ_X_XX_TE 12 1 0.25
python create_basis.py 12
```
The wavefunction is saved under the directory wavefunction. Then going back to the original directory, and run for example,
```
python supervised_NQS.py --l 12 --dim 1 --net MADE --using_complex 1 --Q_tar -1  --lr 1e-3 --opt Adam --path /tmp/ --act relu --alpha_list 4  --T 0.40  --supervised_model 1D_ZZ_1.00X_0.25XX_global_TE_L12  --SP 0 --cost_function joint
```


The general command for running supervised learning is provided below, one can also look into the file `utils/parse_args.py` for detailed information.
```
python supervised_NQS.py --l ${system_size} --dim 1 --net ${net} --using_complex 1 --Q_tar -1  --lr 1e-3 --opt Adam --path ${NN_PATH} --act relu --alpha ${alpha}  --T ${T}  --supervised_model ${supervised_model} --filter_size ${fsize} --num_blocks ${blocks}
```

For comparison with NetKet library, see the directory `netket`.



     

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


### Comparison NetKet
We include the code for comparison using library netket v2. See under directory `netket`.
