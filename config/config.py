import argparse

parser = argparse.ArgumentParser()


def checkbool(x):
    return x.lower() == 'true'


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


# config params
config_arg = add_argument_group('Config Params')
config_arg.add_argument('--config', type=str, default='./config/default_config.json',
                        help='json configuration file, to initialize configs with')

# neural network params
nn_arg = add_argument_group('Neural Network Params')
nn_arg.add_argument('--model_type', type=str,
                    help='whether to use a fully connected <fc> network or a convolutional <conv> network')
nn_arg.add_argument('--fc_params', type=str,
                    help='path to txt file containing number of units per fully connected layer')
nn_arg.add_argument('--conv_params', type=str,
                    help='path to txt file containing convolutional network parameters'
                         'order of arguments is (in_channels, out_channels, kernel_size, stride, padding)')
nn_arg.add_argument('--activation', type=str,
                    help='activation function, one of <sigmoid>, <tanh>, <relu>, <linear>')
nn_arg.add_argument('--norm', type=str,
                    help='whether to use batch norm <batch>, layer norm <layer>, or no norm <none>')
nn_arg.add_argument('--weights_init', type=str,
                    help='weight initialization scheme, one of: pytorch <default>, <xavier_uniform>, <xavier_normal>, <kaiming_uniform>, <kaiming_normal>')

# dataset params
dataset_arg = add_argument_group('Data Params')
dataset_arg.add_argument('--dataset', type=str,
                         help='eiter <MNIST> or <CIFAR10>')

# transforms
transforms_arg = add_argument_group('Transforms Params')
transforms_arg.add_argument('--normalize_input', type=checkbool,
                            help='whether to normalize inputs (using mean & standard deviation of training set)')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--train_random', type=float,
                       help='whether to train on random inputs & random labels,'
                            'when the entropy for random inputs decreases beneath a threshold')
train_arg.add_argument('--epochs', type=int,
                       help='number of epochs to train for')
train_arg.add_argument('--batch_size', type=int,
                       help='batch size during training')
train_arg.add_argument('--optim', type=str,
                       help='<sgd> or <adam>')
train_arg.add_argument('--lr', type=float,
                       help='learning rate')
train_arg.add_argument('--momentum', type=float,
                       help='momentum')
train_arg.add_argument('--nesterov', type=checkbool,
                       help='whether to use nesterov momentum')
train_arg.add_argument('--beta1', type=float,
                       help='beta1 in adam optimizer')
train_arg.add_argument('--beta2', type=float,
                       help='beta2 in adam optimizer')
train_arg.add_argument('--shuffle', type=checkbool,
                       help='whether to shuffle training data')

# database params
db_arg = add_argument_group('Database Params')
db_arg.add_argument('--db_path', type=str,
                    help="path to database file")

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--model_name', type=str,
                      help="model name")
misc_arg.add_argument('--data_dir', type=str,
                      help="directory to load/save dataset from/to")
misc_arg.add_argument('--stdout_dir', type=str,
                      help="directory to log program stdout to")
misc_arg.add_argument('--model_dir', type=str,
                      help='directory in which to save model checkpoints')
misc_arg.add_argument('--num_workers', type=int,
                      help='number of workers/subprocesses to use in dataloader')
misc_arg.add_argument('--device', type=str,
                      help="<cuda> or <cpu>")
misc_arg.add_argument('--random_seed', type=int,
                      help='seed for reproducibility')
misc_arg.add_argument('--save_model', type=checkbool,
                      help='whether to save the model, if validation loss improves, at the end of each epoch')
misc_arg.add_argument('--plot', type=checkbool,
                      help='whether to plot performance metrics')
misc_arg.add_argument('--num_log', type=int,
                      help='number of samples to save additional logs for')
misc_arg.add_argument('--plot_dir', type=str,
                      help='directory in which to save plots')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
