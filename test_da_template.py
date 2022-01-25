from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import time
import socket
import scipy.sparse

from scipy.io import loadmat
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterGrid

from mda_ht_mimic_hs import STEM

import numpy as np
import tensorflow as tf

from generic_utils import random_seed
from generic_utils import data_dir

import multiprocessing as mp
import time
import datetime
import gc
from pprint import pprint
from dataLoader import DataLoader


def main_func(arg_param, src_name=None, trg_name=None):
    print('Running {} ...'.format(os.path.basename(__file__)))

    np.random.seed(random_seed())
    tf.set_random_seed(random_seed())
    tf.reset_default_graph()

    trg_name = arg_param['trg_name']
    if arg_param['data_name'] == "digitFive":
        src_name = "mnist,mnistm,usps,syn,svhn"
        if trg_name == "mnistm":
            arg_param['gen_trade_off'] = 0.1
    else:
        src_name = "caltech,dslr,webcam,amazon"
    src_name_lst = src_name.split(",")
    src_name_lst.remove(trg_name)

    data_loader = DataLoader(src_domain=src_name_lst,
                             trg_domain=trg_name,
                             data_path=arg_param['data_path'],
                             data_format=arg_param['data_format'],
                             cast_data=arg_param['cast_data'],
                             num_class=arg_param['num_classes'])

    assert arg_param['batch_size'] % data_loader.num_src_domain == 0

    learner = STEM(
        **arg_param,
    )

    # learner = create_obj_func(users_params)
    learner.dim_src = data_loader.data_shape
    learner.dim_trg = data_loader.data_shape

    learner.x_trg_test = data_loader.trg_test[0][0]
    learner.y_trg_test = data_loader.trg_test[0][1]

    learner._init(data_loader)
    learner._build_model()
    learner._fit_loop()


def save_data_for_visualization(create_obj_func, src_name=None, trg_name=None, show=False, block_figure_on_end=False):
    print('Running {} ...'.format(os.path.basename(__file__)))

    if src_name is None:
        if len(sys.argv) > 2:
            src_name = sys.argv[2]
        else:
            raise Exception('Not specify source dataset')
    if trg_name is None:
        if len(sys.argv) > 3:
            trg_name = sys.argv[3]
        else:
            raise Exception('Not specify trgget dataset')

    np.random.seed(random_seed())
    tf.set_random_seed(random_seed())
    tf.reset_default_graph()

    print("========== Test on real data ==========")

    users_params = dict()
    users_params = parse_arguments(users_params)

    # data_format = 'libsvm'
    data_format = 'mat'

    if 'format' in users_params:
        data_format, users_params = extract_param('format', data_format, users_params)

    data_loader = DataLoader(src_domain=src_name,
                             trg_domain=trg_name,
                             data_path=data_dir(),
                             data_format=data_format,
                             dataset_name=dataset_name,
                             shuffle_data=True)  # shuffle all training, test data

    assert users_params['batch_size'] % data_loader.num_src_domain == 0

    print('users_params:', users_params)

    learner = create_obj_func(users_params)
    learner.dim_src = data_loader.data_shape
    learner.dim_trg = data_loader.data_shape
    num_tests = 400
    ############## TEST MODEL WITH BALANCED CLASSES OF SRC #####################
    # data_loader.visualize_mini_batch_class_balanced(sample_size=users_params['sample_size'])

    learner.x_trg_test = data_loader.trg_test[0][0]
    learner.y_trg_test = data_loader.trg_test[0][1]
    # learner.x_src_test = x_src_test
    # learner.y_src_test = y_src_test

    print("dim_src: (%d, %d, %d)" % (learner.dim_src[0], learner.dim_src[1], learner.dim_src[2]))
    print("dim_trg: (%d, %d, %d)" % (learner.dim_trg[0], learner.dim_trg[1], learner.dim_trg[2]))

    learner._init(data_loader)
    # learner._build_model()
    learner.visualize_generator_tsne(domain='src', num_tests=num_tests)
    learner.visualize_generator_tsne(domain='trg', num_tests=num_tests)


def run_one_candidate(
        create_obj_func, candidate_params, src_name, trg_name, attribute_names, file_config, num_runs,
        cross, num_features, keep_vars, src_ind_test=None, trg_ind_test=None, grid_search=True, data_format='libsvm'):

    print('OS:', os.name)
    if (os.name == "posix") and file_config is not None:
        print(file_config)
        file = open(file_config, 'r')
        cpu_config = file.read()
        file.close()
        os.system("taskset -p " + cpu_config + " %d" % os.getpid())

    np.random.seed(random_seed())
    tf.set_random_seed(random_seed())
    tf.reset_default_graph()

    train_src_acc_avg = 0
    test_src_acc_avg = 0
    train_trg_acc_avg = 0
    test_trg_acc_avg = 0
    train_time_avg = 0

    log_lst = []
    total_runs = num_runs
    if cross > 0:
        total_runs = num_runs * cross

    load_data_file = load_mat_file_single_label
    if data_format == 'libsvm':
        load_data_file = load_svmlight_file

    for ri in range(total_runs):
        print('----------------------------------')
        if cross > 0:
            if ri % num_runs == 0:
                np.random.seed(1010)
            crossid = int(ri / num_runs)
            print('Run #{0} - Cross #{1}:'.format(ri+1, crossid+1))
            src_train_file_name = os.path.join(data_dir(), src_name + '_' + str(crossid) + '.train.' + data_format)
            src_test_file_name = os.path.join(data_dir(), src_name + '_' + str(crossid) + '.test.' + data_format)
            if src_ind_test is not None:
                src_ind_test_file_name = os.path.join(data_dir(), src_ind_test + '_test.' + data_format)

            trg_train_file_name = os.path.join(data_dir(), trg_name + '_' + str(crossid) + '.train.' + data_format)
            trg_test_file_name = os.path.join(data_dir(), trg_name + '_' + str(crossid) + '.test.' + data_format)
            if trg_ind_test is not None:
                trg_ind_test_file_name = os.path.join(data_dir(), trg_ind_test + '_test.' + data_format)

            print('Train source file:', src_train_file_name)
            print('Valid source file:', src_test_file_name)
            print('Train target file:', trg_train_file_name)
            print('Valid target file:', trg_test_file_name)
            if src_ind_test is not None:
                print(src_ind_test_file_name)
            if trg_ind_test is not None:
                print(trg_ind_test_file_name)
        else:
            print('Run #{0}:'.format(ri+1))
            src_train_file_name = os.path.join(data_dir(), src_name + '_train.' + data_format)
            src_test_file_name = os.path.join(data_dir(), src_name + '_test.' + data_format)
            trg_train_file_name = os.path.join(data_dir(), trg_name + '_train.' + data_format)
            trg_test_file_name = os.path.join(data_dir(), trg_name + '_test.' + data_format)

        if not os.path.exists(src_train_file_name):
            print('File ' + src_train_file_name + 'not found')
            raise Exception('File ' + src_train_file_name + ' not found')
        if not os.path.exists(src_test_file_name):
            raise Exception('File ' + src_test_file_name + ' not found')
        if src_ind_test is not None:
            if not os.path.exists(src_ind_test_file_name):
                raise Exception('File ' + src_ind_test_file_name + 'not found')

        if not os.path.exists(trg_train_file_name):
            print('File ' + trg_train_file_name + 'not found')
            raise Exception('File ' + trg_train_file_name + ' not found')
        if not os.path.exists(trg_test_file_name):
            raise Exception('File ' + trg_test_file_name + ' not found')
        if trg_ind_test is not None:
            if not os.path.exists(trg_ind_test_file_name):
                raise Exception('File ' + trg_ind_test_file_name + 'not found')

        if num_features is None:
            x_src_train, y_src_train = load_data_file(src_train_file_name)
            x_src_test, y_src_test = load_data_file(src_test_file_name)
            if src_ind_test is not None:
                x_src_ind_test, y_src_ind_test = load_data_file(src_ind_test_file_name)

            x_trg_train, y_trg_train = load_data_file(trg_train_file_name)
            x_trg_test, y_trg_test = load_data_file(trg_test_file_name)
            if trg_ind_test is not None:
                x_trg_ind_test, y_trg_ind_test = load_data_file(trg_ind_test_file_name)
        else:
            x_src_train, y_src_train = load_data_file(src_train_file_name, n_features=num_features)
            x_src_test, y_src_test = load_data_file(src_test_file_name, n_features=num_features)
            if src_ind_test is not None:
                x_src_ind_test, y_src_ind_test = load_data_file(src_ind_test_file_name, n_features=num_features)

            x_trg_train, y_trg_train = load_data_file(trg_train_file_name, n_features=num_features)
            x_trg_test, y_trg_test = load_data_file(trg_test_file_name, n_features=num_features)
            if trg_ind_test is not None:
                x_trg_ind_test, y_trg_ind_test = load_data_file(trg_ind_test_file_name, n_features=num_features)

        print('Num source samples: {}'.format(x_src_train.shape[0]))
        print('Num target samples: {}'.format(x_trg_train.shape[0]))

        if grid_search:
            print('Trial params:', dict2string(candidate_params))
        learner = create_obj_func(candidate_params)

        if (data_format == 'libsvm') and (not hasattr(learner, 'sparse') or not learner.sparse):
            x_src_train = x_src_train.toarray()
            x_src_test = x_src_test.toarray()
            if src_ind_test is not None:
                x_src_ind_test = x_src_ind_test.toarray()
            x_trg_train = x_trg_train.toarray()
            x_trg_test = x_trg_test.toarray()
            if trg_ind_test is not None:
                x_trg_ind_test = x_trg_ind_test.toarray()

        print('Training ...')
        learner.dim_src = x_src_train.shape[1:]
        learner.dim_trg = x_trg_train.shape[1:]
        learner.num_classes = len(np.unique(y_src_train))
        learner.fit(None, None, x_src=x_src_train, y_src=y_src_train, x_trg=x_trg_train, y_trg=y_trg_train)

        y_src_train_pred = learner.predict(x_src_train, domain='src')
        y_src_test_pred = learner.predict(x_src_test, domain='src')
        y_trg_train_pred = learner.predict(x_trg_train, domain='trg')
        y_trg_test_pred = learner.predict(x_trg_test, domain='trg')

        src_train_labels, src_train_ycount = np.unique(y_src_train, return_counts=True)
        src_train_acc = metrics.accuracy_score(y_src_train, y_src_train_pred)
        src_train_acc_detail = \
            np.diagonal(metrics.confusion_matrix(y_src_train, y_src_train_pred, src_train_labels)) / src_train_ycount
        trg_train_labels, trg_train_ycount = np.unique(y_trg_train, return_counts=True)
        trg_train_acc = metrics.accuracy_score(y_trg_train, y_trg_train_pred)
        trg_train_acc_detail = \
            np.diagonal(metrics.confusion_matrix(y_trg_train, y_trg_train_pred, trg_train_labels)) / trg_train_ycount

        src_test_labels, src_test_ycount = np.unique(y_src_test, return_counts=True)
        src_test_acc = metrics.accuracy_score(y_src_test, y_src_test_pred)
        src_test_acc_detail = \
            np.diagonal(metrics.confusion_matrix(y_src_test, y_src_test_pred, src_test_labels)) / src_test_ycount
        trg_test_labels, trg_test_ycount = np.unique(y_trg_test, return_counts=True)
        trg_test_acc = metrics.accuracy_score(y_trg_test, y_trg_test_pred)
        trg_test_acc_detail = \
            np.diagonal(metrics.confusion_matrix(y_trg_test, y_trg_test_pred, trg_test_labels)) / trg_test_ycount

        if src_ind_test is not None:
            y_src_ind_test_pred = learner.predict(x_src_ind_test, domain='src')
            src_ind_test_labels, src_ind_test_ycount = np.unique(y_src_ind_test, return_counts=True)
            src_ind_test_acc_detail = \
                np.diagonal(metrics.confusion_matrix(y_src_ind_test, y_src_ind_test_pred, src_ind_test_labels)) \
                / src_ind_test_ycount
            src_ind_test_acc = metrics.accuracy_score(y_src_ind_test, y_src_ind_test_pred)
        if trg_ind_test is not None:
            y_trg_ind_test_pred = learner.predict(x_trg_ind_test, domain='trg')
            trg_ind_test_labels, trg_ind_test_ycount = np.unique(y_trg_ind_test, return_counts=True)
            trg_ind_test_acc_detail = \
                np.diagonal(metrics.confusion_matrix(y_trg_ind_test, y_trg_ind_test_pred, trg_ind_test_labels)) \
                / trg_ind_test_ycount
            trg_ind_test_acc = metrics.accuracy_score(y_trg_ind_test, y_trg_ind_test_pred)

        train_time = learner.train_time
        if grid_search:
            print('Acc on valid source set: {}, Acc on training source set: {}, Training time: {} seconds'
                  .format(src_test_acc, src_train_acc, int(train_time)))
            print('Acc on valid target set: {}, Acc on training target set: {}'
                  .format(trg_test_acc, trg_train_acc))
        else:
            print('Acc on testing source set: {}, Acc on training source set: {}, Training time: {} seconds'
                  .format(src_test_acc, src_train_acc, int(train_time)))
            print('Acc on testing target set: {}, Acc on training target set: {}'
                  .format(trg_test_acc, trg_train_acc))

        train_src_acc_avg += src_train_acc
        test_src_acc_avg += src_test_acc
        train_trg_acc_avg += trg_train_acc
        test_trg_acc_avg += trg_test_acc
        train_time_avg += train_time

        log_lst.append({k: learner.__dict__[k] for k in attribute_names})
        log_lst[len(log_lst) - 1]['src_name'] = src_name
        log_lst[len(log_lst) - 1]['trg_name'] = trg_name
        log_lst[len(log_lst) - 1]['src_train_acc'] = src_train_acc
        log_lst[len(log_lst) - 1]['src_test_acc'] = src_test_acc
        log_lst[len(log_lst) - 1]['trg_train_acc'] = trg_train_acc
        log_lst[len(log_lst) - 1]['trg_test_acc'] = trg_test_acc
        log_lst[len(log_lst) - 1]['train_time'] = train_time
        log_lst[len(log_lst) - 1]['src_train_acc_detail'] = src_train_acc_detail
        log_lst[len(log_lst) - 1]['src_test_acc_detail'] = src_test_acc_detail
        log_lst[len(log_lst) - 1]['trg_train_acc_detail'] = trg_train_acc_detail
        log_lst[len(log_lst) - 1]['trg_test_acc_detail'] = trg_test_acc_detail
        if src_ind_test is not None:
            log_lst[len(log_lst) - 1]['src_independent_test_acc'] = src_ind_test_acc
            log_lst[len(log_lst) - 1]['src_independent_test_acc_detail'] = src_ind_test_acc_detail
        if trg_ind_test is not None:
            log_lst[len(log_lst) - 1]['trg_independent_test_acc'] = trg_ind_test_acc
            log_lst[len(log_lst) - 1]['trg_independent_test_acc_detail'] = trg_ind_test_acc_detail

        for key in keep_vars:
            candidate_params[key] = learner.__dict__[key]

    return \
        train_src_acc_avg / total_runs, test_src_acc_avg / total_runs, \
        train_trg_acc_avg / total_runs, test_trg_acc_avg / total_runs, \
        train_time_avg / total_runs, log_lst, \
        grid_search, candidate_params


train_src_acc_lst = []
test_src_acc_lst = []
train_trg_acc_lst = []
test_trg_acc_lst = []
run_param_lst = []
time_lst = []
testid_lst = []
grid_start_time = None
grid_num_task = None


def log_result(result):
    train_src_acc, test_src_acc, train_trg_acc, test_trg_acc, train_time, log_lst, grid_search, params = result
    train_src_acc_lst.append(train_src_acc)
    test_src_acc_lst.append(test_src_acc)
    train_trg_acc_lst.append(train_trg_acc)
    test_trg_acc_lst.append(test_trg_acc)
    time_lst.append(train_time)
    run_param_lst.append(params)
    testid_lst.append(len(testid_lst) + 1)

    global grid_num_task, grid_start_time
    elapsed_time = time.time() - grid_start_time
    remaining_time = elapsed_time / len(time_lst) * grid_num_task - elapsed_time

    if grid_search:
        print('Acc on valid source set: {}, Acc on training source set: {}, Training time: {} seconds, Params: {}'.
              format(test_src_acc, train_src_acc, int(train_time), dict2string(params)))
        print('Acc on valid target set: {}, Acc on training target set: {}'.
              format(test_trg_acc, train_trg_acc))
    else:
        print("\n========== FINAL RESULT ==========")
        print('Source data: {}'.format(log_lst[len(log_lst) - 1]['src_name']))
        print('Target data: {}'.format(log_lst[len(log_lst) - 1]['trg_name']))
        print('Acc on testing source set: {}, Acc on training source set: {}, Training time: {} seconds'
              .format(test_src_acc, train_src_acc, int(train_time)))
        print('Acc on testing target set: {}, Acc on training target set: {}'
              .format(test_trg_acc, train_trg_acc))

    testid = len(testid_lst)
    log_filename = \
        log_lst[len(log_lst) - 1]['model_name'] + '.' + log_lst[len(log_lst) - 1]['src_name'] \
        + log_lst[len(log_lst) - 1]['trg_name'] \
        + '.' + socket.gethostname() + '.txt'

    log_file = open(log_filename, 'a')
    for it in range(len(log_lst)):
        log_file.write('testid:' + str(testid) + '\trunid:' + str(it) + '\t')
        log_file.write('src_test_acc:' + str(log_lst[it]['src_test_acc']) + '\t')
        log_file.write('src_train_acc:' + str(log_lst[it]['src_train_acc']) + '\t')
        log_file.write('trg_test_acc:' + str(log_lst[it]['trg_test_acc']) + '\t')
        log_file.write('trg_train_acc:' + str(log_lst[it]['trg_train_acc']) + '\t')
        log_file.write('time:' + str(log_lst[it]['train_time']) + '\t')
        for key, value in log_lst[it].items():
            if type(value) is np.ndarray:
                log_file.write(key + ':' + '|'.join('{0:.4f}'.format(x) for x in value.ravel()) + '\t')
            else:
                log_file.write(key + ':' + str(value) + '\t')
        log_file.write('eta:' + str(remaining_time / 3600) + '\t')
        log_file.write('\n')
    log_file.close()


def run_grid_search_multicore(
        create_obj_func, params_gridsearch, attribute_names, src_name=None, trg_name=None,
        num_workers=4, file_config=None,
        num_runs=3, cross=0, num_features=None, full_src_dataset=None, full_trg_dataset=None,
        keep_vars=[], src_ind_test=None, trg_ind_test=None, data_format='libsvm'):
    if src_name is None:
        if len(sys.argv) > 2:
            src_name = sys.argv[2]
        else:
            raise Exception('Not specify source dataset')
    if trg_name is None:
        if len(sys.argv) > 3:
            trg_name = sys.argv[3]
        else:
            raise Exception('Not specify target dataset')

    params_gridsearch = parse_arguments(params_gridsearch, True)

    # print(params_gridsearch)
    file_config, params_gridsearch = extract_param('file_config', file_config, params_gridsearch, scalar=True)
    num_workers, params_gridsearch = extract_param('num_workers', num_workers, params_gridsearch, scalar=True)
    num_runs, params_gridsearch = extract_param('num_runs', num_runs, params_gridsearch, scalar=True)
    cross, params_gridsearch = extract_param('cross', cross, params_gridsearch, scalar=True)
    num_features, params_gridsearch = extract_param('num_features', num_features, params_gridsearch, scalar=True)
    full_src_dataset, params_gridsearch = \
        extract_param('full_src_dataset', full_src_dataset, params_gridsearch, scalar=True)
    full_trg_dataset, params_gridsearch = \
        extract_param('full_trg_dataset', full_trg_dataset, params_gridsearch, scalar=True)
    src_ind_test, params_gridsearch = extract_param('src_ind_test', src_ind_test, params_gridsearch, scalar=True)
    trg_ind_test, params_gridsearch = extract_param('trg_ind_test', trg_ind_test, params_gridsearch, scalar=True)
    data_format, params_gridsearch = extract_param('format', data_format, params_gridsearch, scalar=True)

    if src_ind_test is not None:
        if full_src_dataset is None:
            src_ind_test = src_name
        else:
            src_ind_test = full_src_dataset
    if trg_ind_test is not None:
        if full_trg_dataset is None:
            trg_ind_test = trg_name
        else:
            trg_ind_test = full_trg_dataset

    if full_src_dataset is None:
        full_src_dataset = src_name
    if full_trg_dataset is None:
        full_trg_dataset = trg_name

    if not os.path.exists(os.path.join(data_dir(), full_src_dataset + '_train.' + data_format)):
        print('File' + full_src_dataset + '_train.' + data_format + ' is not found')
    if not os.path.exists(os.path.join(data_dir(), full_trg_dataset + '_train.' + data_format)):
        print('File' + full_trg_dataset + '_train.' + data_format + ' is not found')

    candidate_params_lst = list(ParameterGrid(params_gridsearch))
    grid_search = True
    if len(candidate_params_lst) == 1:
        grid_search = False

    global grid_start_time, grid_num_task
    grid_start_time = time.time()
    grid_num_task = len(candidate_params_lst)

    pool = mp.Pool(num_workers)  # maximum of workers
    result_lst = []
    for candidate_params in candidate_params_lst:
        result = pool.apply_async(
            run_one_candidate,
            args=(
                create_obj_func, candidate_params, src_name, trg_name, attribute_names, file_config, num_runs, cross,
                num_features, keep_vars, src_ind_test, trg_ind_test, grid_search, data_format),
            callback=log_result
        )
        result_lst.append(result)

    for result in result_lst:
        result.get()
    pool.close()
    pool.join()

    # if len(candidate_params_lst) > 1:
    print("========== FINAL RESULT ==========")
    idx_best = np.argmax(np.array(test_trg_acc_lst))
    print('Source: {}'.format(src_name))
    print('Target: {}'.format(trg_name))
    print('Best testid: {}'.format(testid_lst[idx_best]))
    print('Best acc on training source set: {}'.format(train_src_acc_lst[idx_best]))
    print('Best acc on valid source set: {}'.format(test_src_acc_lst[idx_best]))
    print('Best acc on training target set: {}'.format(train_trg_acc_lst[idx_best]))
    print('Best acc on valid target set: {}'.format(test_trg_acc_lst[idx_best]))
    print('Best params: {}'.format(run_param_lst[idx_best]))

    if cross > 0:
        print('Run the best one')
        num_runs_for_best = num_runs
        if num_runs < 3:
            num_runs_for_best = 3
        best_result = run_one_candidate(
            create_obj_func, run_param_lst[idx_best], full_src_dataset, full_trg_dataset,
            attribute_names, file_config, num_runs_for_best,
            cross=0, num_features=num_features, keep_vars=keep_vars)
        # best_result['gridsearch_time'] = np.sum(np.array(time_lst))
        log_result(best_result)


# def main_func(
#         create_obj_func,
#         choice_default=0,
#         src_name_default='svmguide1',
#         trg_name_default='svmguide1',
#         run_exp=False,
#         keep_vars=[],
#         **kwargs):
#
#     src_name = src_name_default
#     trg_name = trg_name_default
#
#     test_real_dataset(create_obj_func, src_name, trg_name)


def parse_arguments(params, as_array=False):
    for it in range(4, len(sys.argv), 2):
        params[sys.argv[it]] = parse_argument(sys.argv[it + 1], as_array)
    return params


def parse_argument(string, as_array=False):
    try:
        result = int(string)
    except ValueError:
        try:
            result = float(string)
        except ValueError:
            if str.lower(string) == 'true':
                result = True
            elif str.lower(string) == 'false':
                result = False
            elif string == "[]":
                return []
            elif ('|' in string) and ('[' in string) and (']' in string):
                result = [float(item) for item in string[1:-1].split('|')]
                return result
            elif (',' in string) and ('(' in string) and (')' in string):
                split = string[1:-1].split(',')
                result = float(split[0]) ** np.arange(float(split[1]), float(split[2]), float(split[3]))
                return result
            else:
                result = string

    return [result] if as_array else result


def resolve_conflict_params(defaul_params, additional_params):
    '''
        remove duplicate param in additinal_param
    '''
    for key in defaul_params.keys():
        if key in additional_params.keys():
            del additional_params[key]
    return additional_params

def extract_param(key, value, params_gridsearch, scalar=False):
    if key in params_gridsearch.keys():
        value = params_gridsearch[key]
        del params_gridsearch[key]
        if scalar and (value is not None):
            value = value[0]
    return value, params_gridsearch


def dict2string(params):
    result = ''
    for key, value in params.items():
        if type(value) is np.ndarray:
            if value.size < 16:
                result += key + ': ' + '|'.join('{0:.4f}'.format(x) for x in value.ravel()) + ', '
        else:
            result += key + ': ' + str(value) + ', '
    return '{' + result[:-2] + '}'


def load_mat_file_single_label(filename):
    filename_list = ['mnist', 'stl32', 'synsign', 'gtsrb', 'cifar32', 'usps32']
    data = loadmat(filename)
    x = data['X']
    y = data['y']
    if any(fn in filename for fn in filename_list):
        if 'mnist32_60_10' not in filename and 'mnistg' not in filename:
            y = y[0]
        else:
            y = np.argmax(y, axis=1)
    # process one-hot label encoder
    elif len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    return x, y


def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    return x.astype('float32') / 255 * 2 - 1
