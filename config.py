import argparse

parser = argparse.ArgumentParser()


def parsebool(x):
    return x.lower() == 'true'


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


# neural network params
nn_arg = add_argument_group('Neural Network Params')
nn_arg.add_argument('--nn_type', type=str, default='fc',
                    help='whether to use a convolutional network or a fully connected network')
nn_arg.add_argument('--fc_params', nargs='+', type=int, default=[1000, 1000],
                    help='number of units per fully connected layer')
nn_arg.add_argument('--conv_params', default=
                                             [
                                                 (3, 128, 9, 1, 0),
                                                 (128, 128, 7, 1, 0),
                                                 (128, 256, 5, 1, 0),
                                                 (256, 512, 5, 1, 0)
                                             ],
                    help='convolutional network parameters'
                         'order of arguments is (in_channels, out_channels, kernel_size, stride, padding)')
nn_arg.add_argument('--activation', type=str, default='relu',
                    help='activation function, one of <sigmoid>, <tanh>, <relu>')
nn_arg.add_argument('--norm', type=str, default='',
                    help='whether to use batch norm <batch>, layer norm <layer>, or no norm <>')

# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--dataset', type=str, default='MNIST',
                      help='eiter <MNIST> or <CIFAR10>')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=15,
                       help='number of epochs to train for')
train_arg.add_argument('--batch_size', type=int, default=128,
                       help='batch size during training')
train_arg.add_argument('--use_sgd', type=parsebool, default=False,
                       help='whether to use sgd <True> or adam <False> optimizer')
train_arg.add_argument('--lr', type=float, default=3e-4,
                       help='learning rate')
train_arg.add_argument('--momentum', type=float, default=0.9,
                       help='momentum')
train_arg.add_argument('--use_nesterov', type=parsebool, default=True,
                       help='whether to use nesterov momentum')
train_arg.add_argument('--beta1', type=float, default=0.9,
                       help='beta1 in adam optimizer')
train_arg.add_argument('--beta2', type=float, default=0.999,
                       help='beta2 in adam optimizer')
train_arg.add_argument('--shuffle', type=parsebool, default=True,
                       help='whether to shuffle training data')
train_arg.add_argument('--num_workers', type=int, default=4,
                       help='number of workers/subprocesses to use in dataloader')

# adversarial robustness params
# for evaluation only i.e. not for adversarial training
adversary_arg = add_argument_group('Adversarial Robustness Params')
adversary_arg.add_argument('--attack_type', type=str, default='Linf',
                           help='adversarial attack model')
adversary_arg.add_argument('--eps', type=float, default=0.1,
                           help='epsilon')
adversary_arg.add_argument('--nb_iter', type=int, default=20,
                           help='number of iterations for adversary')
adversary_arg.add_argument('--eps_iter', type=float, default=0.01,
                           help='eps_iter')
adversary_arg.add_argument('--targeted', type=parsebool, default=False,
                           help='targeted or untargeted attack')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--model_name', type=str, default='',
                      help="model name")
misc_arg.add_argument('--stdout_dir', type=str, default='./stdout',
                      help="directory to log program stdout to")
misc_arg.add_argument('--model_dir', type=str, default='./ckpt',
                      help='directory in which to save model checkpoints')
misc_arg.add_argument('--use_gpu', type=parsebool, default=True,
                      help="whether to use a gpu, if available")
misc_arg.add_argument('--random_seed', type=int, default=2,
                      help='seed for reproducibility')
misc_arg.add_argument('--save_model', type=parsebool, default=True,
                      help='whether to save the model, if validation loss improves, at the end of each epoch')
misc_arg.add_argument('--plot', type=parsebool, default=True,
                      help='whether to plot performance metrics')
misc_arg.add_argument('--plot_dir', type=str, default='./plots',
                      help='directory in which to save plots')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
