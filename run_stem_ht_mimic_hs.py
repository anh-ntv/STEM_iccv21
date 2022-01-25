from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from layers import noise
from test_da_template import main_func, resolve_conflict_params

from tensorflow.python.layers.core import dropout
from tensorbayes.layers import dense, conv2d, avg_pool, max_pool

import warnings
import os
from generic_utils import tuid, model_dir
import signal
import sys
import time
import datetime
from pprint import pprint
import param_config
import yaml

model_name = "MDAOT-results"
current_time = tuid()
# generator
def encode_layout(preprocess, training_phase=True, data_name='digitFive'):
    layout = []
    if data_name == 'digitFive':
        layout = [
            (preprocess, (), {}),  # process image data before using convolutional neural network
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (max_pool, (2, 2), {}),
            (dropout, (), dict(training=training_phase)),
            (noise, (1,), dict(phase=training_phase)),
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (max_pool, (2, 2), {}),
            (dropout, (), dict(training=training_phase)),
            (noise, (1,), dict(phase=training_phase)),
        ]
    elif data_name == 'officeCaltech10':
        layout = [
            (dense, (256,), {}),
            (dropout, (), dict(training=training_phase)),
            (noise, (1,), dict(phase=training_phase)),
        ]
    return layout


# classifier
def class_discriminator_layout(num_classes=None, global_pool=True, activation=None, data_name='digitFive'):
    layout = []
    if data_name == 'digitFive':
        layout = [
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (conv2d, (64, 3, 1), {}),
            (avg_pool, (), dict(global_pool=global_pool)),
            (dense, (num_classes,), dict(activation=activation))
        ]

    elif data_name == 'officeCaltech10':
        layout = [
            (dense, (num_classes,), dict(activation=activation))
        ]
    return layout


# source discriminator
def domain_layout(c, data_name='digitFive'):
    layout = []
    if data_name == 'digitFive':
        layout = [
            (dense, (100,), {}),
            (dense, (c,), dict(activation=None))
        ]
    elif data_name == 'officeCaltech10':
        layout = [
            (dense, (c,), dict(activation=None))
        ]
    return layout


# discriminator src and trg
def disc_layout(c, data_name='digitFive'):
    layout = []
    if data_name == 'digitFive':
        layout = [
            (dense, (100,), {}),
            (dense, (c,), dict(activation=None))
        ]
    elif data_name == 'officeCaltech10':
        layout = [
            (dense, (c,), dict(activation=None))
        ]
    return layout

def merge_params(defaul_params, additional_params):
    '''
        replace and merge param in additinal_param to defaul param
    '''
    merged_param = defaul_params.copy()
    for key in additional_params.keys():
        merged_param[key] = additional_params[key]
    return merged_param


def main_test():
    arg_param = param_config.args
    with open(r"./config/{}".format(arg_param['config'])) as file:
        configs = yaml.full_load(file)

    arg_param['encode_layout'] = encode_layout
    arg_param['classify_layout'] = class_discriminator_layout
    arg_param['domain_layout'] = domain_layout
    arg_param['disc_layout'] = disc_layout

    merged_params = merge_params(arg_param, configs)
    main_func(merged_params)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.console_log_path = os.path.join(model_dir(), model_name, "console_output", "{}.txt".format(current_time))
        if not os.path.exists(os.path.dirname(self.console_log_path)):
            os.makedirs(os.path.dirname(self.console_log_path))
        self.log = open(self.console_log_path, 'a')
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C.')
        self.log.close()

        # Remove logfile
        # os.remove(self.console_log_path)
        print('Save log file at:', self.console_log_path)
        sys.exit(0)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == '__main__':
    # pytest.main([__file__])
    sys.stdout = Logger()
    start_time = time.time()
    print('Running {} ...'.format(os.path.basename(__file__)))
    main_test()
    training_time = time.time() - start_time
    print('Total time: %s' % str(datetime.timedelta(seconds=training_time)))
    print("============ LOG-ID: %s ============" % current_time)
