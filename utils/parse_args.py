import argparse
import sys


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description='Variational Monte Carlo with NNQS')
    parser.add_argument('--l', dest='L',
                        help='system size. Default: 10',
                        default=10, type=int)
    parser.add_argument('--net', dest='which_net',
                        help='Name of the Neural Network. Default: sRBM',
                        default='sRBM', type=str)
    parser.add_argument('--lr', dest='lr',
                        help='learning rate. Default: 1e-3',
                        default=1e-3, type=float)
    parser.add_argument('--num_iter', dest='num_iter',
                        help='number of iteration for optimization. It is suggested'
                        ' that the multiplication of learning rate and number of iteration'
                        ' be around one, i.e. lr * num_iter = 1.  Default: 1500',
                        default=1500, type=int)
    parser.add_argument('--num_sample', dest='num_sample',
                        help='Number of sampling in Monte Carlo process. Default: 5000',
                        default=5000, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in Network pretraining. Default: 500',
                        default=500, type=int)
    parser.add_argument('--alpha', dest='alpha',
                        help='controll parameter for model complexity. Default: 4',
                        default=4, type=int)
    parser.add_argument('--opt', dest='opt',
                        help='optimizer for the neural network. Default: Momentum method',
                        default='Mom', type=str)
    parser.add_argument('--H', dest='H',
                        help='target Hamiltonian for optimization. Default: AFH',
                        default='AFH', type=str)
    parser.add_argument('--dim', dest='dim',
                        help='Dimension of the system. 1d: chain, 2d: square lattice.'
                        'Input should be integer. Default: 1',
                        default=1, type=int)
    parser.add_argument('--J2', dest='J2',
                        help='The J2/J1 value in J1J2 model'
                        ' Default: 1.',
                        default=1., type=float)
    parser.add_argument('--SR', dest='SR',
                        help='Using Stochastic Reconfiguration (SR) method or not.'
                        ' Giving True(1) or False(0). Default: 1.',
                        default=1., type=int)
    parser.add_argument('--reg', dest='reg',
                        help='Scaling factor for fixed scale weight decay for regularization'
                        ', s.t. gradient +=  scale * W.  Default: 0.',
                        default=0., type=float)
    parser.add_argument('--path', dest='path',
                        help='path to the directory where wavefunction and E_log are saved. '
                        'Default: \'\'',
                        default='', type=str)
    parser.add_argument('--act', dest='act',
                        help='nonlinear activation function in the network'
                        'Default: softplus2',
                        default='softplus2', type=str)
    parser.add_argument('--SP', dest='SP',
                        help='True(1): single precision, False(0): double precision '
                        'Default: True(1)',
                        default='1', type=int)
    parser.add_argument('--using_complex', dest='using_complex',
                        help='False(0): using real-valued wavefunction.'
                        'True(1): using complex-valued wavefunction. '
                        'Default: False(0)',
                        default='0', type=int)






    if len(sys.argv) == 1:
        pass
        # parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args
