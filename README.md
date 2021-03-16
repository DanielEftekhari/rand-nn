# Improving Generalization by Training on Random Noise #

**Requirements**:

Python 3.5 or higher (tested on Python 3.6).<br />

Specific module and version requirements are listed in requirements.txt. After cloning the repository,<br />
cd to the repo directory, and enter `pip3 install -r requirements.txt`<br />
Note: You may need to run `apt-get install python3-tk` afterwards to have matplotlib work correctly.

**File Descriptions**:

`train.py` -> training call logic - the only file that needs to be executed<br />
`models.py` -> the fully connected and convolutional networks are defined here<br />
`activations.py` -> activation functions<br />
`layers.py` -> custom layer utils, including layer normalization<br />
`loss_fns.py` -> custom cross entropy loss function; generalizes the cross entropy loss to cases where the label is a probability distribution over the targets, rather than a one-hot encoding. needed when training on random inputs<br />
`plotting.py` -> plotting utils<br />
`utils.py` -> miscellaneous utils<br />

`./config/config.py` -> complete list of command-line parsable arguments<br />
`./config/default_config.json` -> default config parameters; arguments provided in the command line override the default parameters<br />
`./db/dbTiny.py` -> various wrapper functions to facilitate interfacing with tinydb, without needing knowledge of the package<br />
`./db/db.json` -> a json file which contains all the config parameters used for the run, and various model performance statistics. this is meant to facilitate tracking & comparison of different experiments, especially when the number of experiments grows very large<br />
`./params/fc_params.txt` -> default fully connected network architecture; see config.json for specification details<br />
`./params/conv_params.txt` -> default convolutional network architecture; see config.json for specification details<br />

**Usage**:

To train and validate a fully connected architecture on the MNIST classification task, enter `python3 train.py`. See config.py for the list of arguments which can be passed in,
including passing the neural network architecture as an argument (by default it is set to a ReLU activated neural network with 2 hidden layers).

To train on random inputs when the entropy of the softmax distribution on random inputs decreases below a threshold, enter `python3 train.py --train_random=<x>`, where `<x>` is in the range [0, 1].
