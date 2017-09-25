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
                        help='Name of the Neural Network. Default: NN',
                        default='NN', type=str)
    parser.add_argument('--lr', dest='lr',
                        help='learning rate. Default: 1e-3',
                        default=1e-3, type=float)
    parser.add_argument('--num_sample', dest='num_sample',
                        help='Number of sampling in Monte Carlo process. Default: 500',
                        default=500, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='Batch size in Network pretraining. Default: 100',
                        default=100, type=int)
    parser.add_argument('--alpha', dest='alpha',
                        help='controll parameter for model complexity. Default: 4',
                        default=4, type=int)
    parser.add_argument('--opt', dest='opt',
                        help='optimizer for the neural network. Default: Momentum method',
                        default='Mom', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
