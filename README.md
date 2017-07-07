# Neural Network Quantum State (NNQS)

The project aim to treat neural network as an tensor approximation tool by training the neural network with one regression output, i.e. the tensor value.

The original motivation is to study the obtain the ground state of quantum many body problems, where the ground state is a high dimensional tensor.


### Requirements

  - python 2.7
  - numpy, tensorflow



### Reference

This work is inspired by the paper from,

* Giuseppe Carleo, Matthias Troyer - [Solving the Quantum Many-Body Problem with Artificial Neural Networks](https://arxiv.org/abs/1606.02318) 
* James Martens, et al. - [On the Representational Efficiency of Restricted Boltzmann Machines](https://www.cs.toronto.edu/~toni/Papers/nips2013.pdf)

### Execute Code

to run the variational monte carlo,
```
python NQS.py --net [which_net] --l [L] --lr [learning_rate] --num_sample [num_monte_carlo_sampling]
```
to run the pretraining, 
```
python pretrain.py --net [which_net] --l [L] 
```
L is the system size, which_net should be the network name.
For example
```
python NQS.py --net NN3 --l 20 --lr 1e-3 --num_sample 500
```


###   Network Architecture

* **1-hidden layer NN (NN)**   
*input_layer -> affine -> tanh -> affine ->  output_layer*
* **3-hidden layer NN (NN3)**   
*input_layer -> [affine -> tanh] x 3  -> affine ->  output_layer*
* **1 conv +  1-hidden layer NN (CNN)**   
*input_layer -> conv -> leaky_relu -> affine -> tanh -> affine -> output_layer*
* **Fully Convolutional Network (FCN)**    
*input_layer -> [conv -> leaky_relu -> conv -> leaky_relu -> avg_pool ] x 4 -> affine -> output_layer*
* **1-hidden layer Complex NN (NN_complex)**    
*input_layer -> complex_affine -> exp -> complex_affine --> output_layer*
* **3-hidden layer Complex NN (NN3_complex)**    
*input_layer -> [affine -> sigmoid] x 2 -> complex_affine -> exp -> complex_affine -> output_layer*
* **RBM Network (NN_RBM)**    
*input_layer -> complex_affine -> soft_plus -> sum -> 1*    
*input_layer -> affine -> 2*    
*1+2 -> exp -> output_layer*

### Table of pretraining accuracy 
Learning to approximate AFH ground state for system size L = 16, dim(H) = 65536,

| Network Name  | number of parameters  | accuracy on AFH  | iterations to converge (learning rate)  |
| --- | --- | --- | --- |
| NN (alpha=9) | 2593  |  57 %  | > 70k (1e-2) |
| NN3 (alpha=2)  | 2688  | 99 %  | > 100k (1e-2) |
| CNN  |  1791 | 50 %  | > 150k (1e-3) | 
| FCN  | 1648  | 53 %  | > 100k (1e-2) |
| FCN  | 1648  | 65 %  | 200k (1e-3) |
| NN_complex  | 1054  | 73 %  | 60 k (1e-2) |
| NN_complex  | 1054  | 71 %  | 200k (1e-3) |
| NN3_complex  | 1121  | 99 %  | 50 k (1e-2 -> 1e-3) |
| NN_RBM  | 1104  | 99 % | ~ 100 k (1e-2) |


NN3 : \[45k -> 85%\] but keep oscillating +- 10%  
NN3_complex : \[20k -> 90%\] \[30k -> 97%\]  
RBM : [25k -> 95%] quite stable   
After 100k, reducing the step size, NN3, NN_RBM both quickly gives 99% accuracy.

>max accuracy :  
RBM 99.9%  
NN3_complex 99.7%

>Observation:  
relu Conv arch, need smaller lr. The other don't


### To do list

- [x] Variational Monte Carlo (VMC) with NNQS
- [x] Pretraining with eigenvector given.
- [ ] Different NNQS
     - [x] 1-hidden layer NN
     - [x] 3-hidden layer NN
     - [x] 1 conv +  1-hidden layer NN
     - [x] Fully Convolutional Network (FCN)
     - [x] 1-hidden layer Complex NN
     - [x] 3-hidden layer Complex NN
     - [x] RBM Network
     - [ ] ...
- [ ] Test Pretrain Accuracy
- [ ] Test VMC Accuracy




