# Neural Network Quantum State (NNQS)

The project aim to treat neural network as an tensor approximation tool by training the neural network with one regression output, i.e. the tensor value.

The original motivation is to study the obtain the ground state of quantum many body problems, where the ground state is a high dimensional tensor.


### Requirements

  - python 2.7 or >= 3
  - numpy, tensorflow 1.4     
  Note that with tensorflow <= 1.3, Jastrow wavefunction seems not to work properly.



### Reference

This work is inspired by the paper from,

* Giuseppe Carleo, Matthias Troyer - [Solving the Quantum Many-Body Problem with Artificial Neural Networks](https://arxiv.org/abs/1606.02318) 
* James Martens, et al. - [On the Representational Efficiency of Restricted Boltzmann Machines](https://www.cs.toronto.edu/~toni/Papers/nips2013.pdf)
* C.P. Chou, et al. - [Matrix-Product based Projected Wave Functions Ansatz for Quantum Many-Body Ground States](https://arxiv.org/abs/1201.6121)
* Olga Sikora, et al. - [Variational Monte Carlo simulations using tensor-product projected states](https://arxiv.org/abs/1407.4107)

### Github Reference

* [PixelCNN](https://github.com/israelg99/eva/blob/master/eva/util/mutil.py)

### How to run the Code

To run the variational monte carlo,
```
python train_NQS.py --net [which_net] --dim [dim] --l [L] --lr [learning_rate] --num_sample [num_monte_carlo_sampling]
```
L is the system size, which_net should be the network name.
For example
```
python train_NQS.py --net sRBM --dim 1 --l 20 --num_sample 1000 --batch_size 200 --H AFH
```
To see detailed of the input arguments, run
```
python train_NQS.py --help
```
 

To run the pretraining, one first need to provide the exact eigenstate in the EigenVec folder, (I will add python code in the future) 
```
python pretrain.py --net [which_net] --l [L] 
```


### Result

#### 1d J1J2
![please let me know if the figures are not displayed on the website.](/Figures/1dvmc.png)
In 1d the result does not seem to be good for having multiple layers, especially after the state is dimerized.

#### 2d J1J2
![please let me know if the figures are not displayed on the website.](/Figures/2dvmc.png)
 In 2d the result seems to be good for FCN2.

###   Network Architecture

Relevent codes are in /network/tf_network.py   
* **1-hidden layer NN version1(NN_v1)**   
*input_layer -> affine -> tanh -> affine ->  output_layer*
* **1-hidden layer NN version2(NN_v2)**   
*input_layer -> affine -> tanh -> (1)*   
*(1) -> affine ->  exp -> out1*  
*(1) -> affine ->  cos -> out2*   
*out1 x out2 -> output_layer*     
* **3-hidden layer NN (NN3)**   
*input_layer -> [affine -> tanh] x 3  -> affine ->  output_layer*
* **1 conv +  1-hidden layer NN (CNN)**   
*input_layer -> conv -> leaky_relu -> affine -> tanh -> affine -> output_layer*
* **Fully Convolutional Network (FCN)**    
*input_layer -> [conv -> leaky_relu -> conv -> leaky_relu -> avg_pool ] x 4 -> affine -> output_layer*
* **1-hidden layer Complex NN version1 (NN_complex_v1)**    
*input_layer -> complex_affine -> exp -> complex_affine -> output_layer*
* **1-hidden layer Complex NN version2 (NN_complex_v2)**    
*input_layer -> complex_affine -> soft_plus -> complex_affine -> exp -> real -> output_layer*
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
| NN_v1 (alpha=9) | 2593  |  57 %  | > 70k (1e-2) |
| NN_v2 (alpha=4) | 1218  |  99 %  | 50k (1e-2 -> 1e-3) |
| NN3 (alpha=2)  | 2688  | 99 %  | > 100k (1e-2) |
| CNN  |  1791 | 50 %  | > 150k (1e-3) | 
| FCN  | 1648  | 65 %  | 200k (1e-3) |
| NN_complex_v1  | 1153  | 73 %  | 60 k (1e-2) |
| NN_complex_v2  | 1153  | 99% | 35k (1e-3) |
| NN3_complex  | 1121  | 99 %  | 50 k (1e-2 -> 1e-3) |
| NN_RBM  | 1104  | 99 % | ~ 100 k (1e-2) |



NN3 : \[45k -> 85%\] but keep oscillating +- 10%  
NN_complex : [20k -> 95%]  
NN3_complex : \[20k -> 90%\] \[30k -> 97%\]  
RBM : [25k -> 95%] quite stable   
After 100k, reducing the step size, NN3, NN_RBM both quickly gives 99% accuracy.


>max accuracy :  
RBM: 99.9%  
NN_complex: 99.8%   
NN3_complex: 99.7%   
NN_v2:  99.6%

>Observation:   
>1. Soft_plus + complex network works very well. Relu + complex network works not so well. This is probably because the inputs for activation are small, i.e. [-1,1]. In orther words, input are not at asymptotic region, where Soft_plus and relu are almost the same. 
>2. Exponential is crucial for complex network.
>3. relu Conv arch, need smaller lr. Network involving exponential also need samller lr.  
>4. Standard neural network can also represent the wave function. However, the convergence is slower. This might indicate that it is hard to obtain ground state in VMC optimization, even though the function space contain the wave function.
>5. Real shallow neural network does not work.
>6. Complex network is equivalent to a real network with learning phase and magnitude seperately. The network can share feature until the final layer which output the phase and magnitude seperately.


### Benchmark

ED:    
L=10 AFH PBC: -4.515446e-01    
L=16 AFH PBC: -4.463935e-01       
L=10 AFH OBC: -4.258035e-01   
L=16 AFH OBC: -4.319836e-01     
2d 4x4 AFH PBC: -0.7017802   

DMRG bond-dim40:   
L=40 AFH OBC: -4.3853682e-01   
L=40 AFH PBC: -4.4366e-01    
   
2d QMC
AFH PBC: -   

### To do list

- [x] Variational Monte Carlo (VMC) with NNQS
- [x] Pretraining with eigenvector given.
- [x] Different NNQS
     - [x] 1-hidden layer NN
     - [x] 3-hidden layer NN
     - [x] 1 conv +  1-hidden layer NN
     - [x] Fully Convolutional Network (FCN)
     - [x] 1-hidden layer Complex NN
     - [x] 3-hidden layer Complex NN
     - [x] RBM Network
     - [x] Jastrow wavefunction (2d)
- [ ] Test Pretrain Accuracy
- [ ] Neural Network Projected Quantum State (NNPQS)
- [x] Test VMC Accuracy
     - [x] NNQS 1d AFH
     - [x] NNQS 2d AFH
     - [x] NNQS 1d J1J2
     - [x] NNQS 2d J1J2
     - [ ] NNPQS 
- [ ] Properties
     - [ ] Scaling behavior of the number of parameters 
     - [ ] Entanglement structure
     - [ ] transfer learning ( known WF or smaller system WF )
     
