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
                        default=500, type=int)
    parser.add_argument('--save_each', dest='save_each',
                        help='number of iterations that wavefunction and results'
                        'are stored. Default: 100',
                        default=100, type=int)
    parser.add_argument('--num_sample', dest='num_sample',
                        help='Number of sampling in Monte Carlo process. Default: 4096',
                        default=4096, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in Network pretraining. Default: 512',
                        default=512, type=int)
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
    parser.add_argument('--real_time', dest='real_time',
                        help='False(0): using imaginary time evolution'
                        'True(1): using real time evolution '
                        'Default: False(0)',
                        default='0', type=int)
    parser.add_argument('--integration', dest='integration',
                        help='numerical integration method'
                        'Default: \'rk4\'',
                        default='rk4', type=str)
    parser.add_argument('--pinv_rcond', dest='pinv_rcond',
                        help='Cutoff for small singular values,'
                        'in the routine of np.linalg.pinv'
                        'Default: 1e-6',
                        default=1e-6, type=float)
    parser.add_argument('--debug', dest='debug',
                        help='flag for debug, the seed for random number'
                        'is fixed, so the result would be reproducable.'
                        ' Giving True(1) or False(0). Default: 0.',
                        default=0, type=int)
    parser.add_argument('--PBC', dest='PBC',
                        help='Determine the boundary condition;'
                        'True if PBC, False if OBC. Default: False(0)',
                        default=0, type=int)
    parser.add_argument('--num_blocks', dest='num_blocks',
                        help='Determine the number of block in pixelCNN sturcture;'
                        'Default: 10',
                        default=10, type=int)
    parser.add_argument('--multi_gpus', dest='multi_gpus',
                        help='Using multiple gpus or not'
                        'Default: False',
                        default=0, type=int)
    parser.add_argument('--conserved_C4', dest='conserved_C4',
                        help='whether having C4 conservation'
                        'Default: False',
                        default=0, type=int)
    parser.add_argument('--conserved_inv', dest='conserved_inv',
                        help='whether having spin inversion symmetry'
                        'Default: False',
                        default=0, type=int)
    parser.add_argument('--conserved_Sz', dest='conserved_Sz',
                        help='whether having charge conservation'
                        'Default: True',
                        default=1, type=int)
    parser.add_argument('--conserved_SU2', dest='conserved_SU2',
                        help='whether having SU2 conservation'
                        'Default: False',
                        default=0, type=int)
    parser.add_argument('--warm_up', dest='warm_up',
                        help='whether to using warm up'
                        'Default: False',
                        default=0, type=int)
    parser.add_argument('--Q_tar', dest='Q_tar',
                        help='the target conserved charge/spin'
                        'must specify value if conserved_Sz is True'
                        'and only affect when conserved_Sz is True'
                        'Default: None',
                        default=None, type=int)
    parser.add_argument('--chem_pot', dest='chem_pot',
                        help='the value of chemical potential'
                        'if not not using set to None'
                        'Default: None',
                        default=None, type=float)




    if len(sys.argv) == 1:
        pass
        # parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args
