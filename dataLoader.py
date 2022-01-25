import os

import numpy as np
from scipy.io import loadmat
from scipy import misc
import cv2
import time
import h5py
from keras.utils.np_utils import to_categorical
from generic_utils import random_seed
import csv

dataset_name_map = {
    'mnistm': ['digits', 'mnistm32_60_10'],
    'mnist': ['digits', 'mnist32_60_10'],
    'svhn': ['digits', 'svhn'],
    'syn': ['digits', 'syn32'],
    'usps': ['digits', 'usps32'],
    'caltech': ['office_caltech10_ResNet101_feat', 'caltech_ResNet101'],
    'dslr': ['office_caltech10_ResNet101_feat', 'dslr_ResNet101'],
    'webcam': ['office_caltech10_ResNet101_feat', 'webcam_ResNet101'],
    'amazon': ['office_caltech10_ResNet101_feat', 'amazon_ResNet101']
}


def shuffle(x, y=None, r_seed=random_seed()):
    np.random.seed(r_seed)
    idx_train = np.random.permutation(x.shape[0])
    x = x[idx_train]
    if y is not None:
        y = y[idx_train]
    return x, y


def load_mat_office_caltech10_decaf(filename):
    data = loadmat(filename)
    x = np.reshape(data['feas'], (-1, 8, 8, 64))
    # x = np.reshape(data['feas'], (-1, 8, 8, 32))

    # x = data['feas']
    y = data['labels']
    y = y.reshape(-1) - 1
    return x, y


def load_office31_resnet50_feature(file_path_train):
    file = open(file_path_train, "r")
    reader = csv.reader(file)
    features_full = []
    labels_full = []
    for line in reader:
        feature_i = np.asarray(line[:2048]).astype(np.float32)
        label_i = int(float(line[2048]))
        features_full.append(feature_i)
        labels_full.append(label_i)
    features_full = np.asarray(features_full)
    labels_full = np.asarray(labels_full)
    return features_full, labels_full


def load_mat_office_caltech10_ResNet101(filename, tail_name="_train"):
    data = loadmat(filename)
    # x = np.reshape(data['feas'], (-1, 8, 8, 32))

    x = data['feas']
    y = data['labels'][0]

    x, y = shuffle(x, y)

    train_rate = 0.8
    num_train = int(len(x) * train_rate)

    if tail_name == "_train":
        x = x[:num_train]
        y = y[:num_train]
    else:
        x = x[num_train:]
        y = y[num_train:]
    return x, y


def load_mat_domain_net_ResNet101(filename, num_load=None, poiter=0, isshuffle=False, tail="_train", num_class=10):
    data = loadmat(filename)
    # x = np.reshape(data['feas'], (-1, 8, 8, 32))

    x = np.asarray(data['feas'])
    y = np.asarray(data['labels'][0])

    # s = ""
    # classes, c = np.unique(y, return_counts=True)
    # for uni_y, num_per in zip(classes, c):
    #     s += "\tclass {}\t{}".format(uni_y, num_per)
    # print(s)
    # n = filename.split("/")[-1].split(".")[0]
    # plt.figure("{}.png".format(n))
    # plt.hist(y)
    # plt.savefig()

    if num_class is not None:
        idx_class = np.where(y < num_class)
        y = y[idx_class]
        x = x[idx_class]

    # train_rate = 0.7
    # num_train = int(len(x) * train_rate)
    # if tail == "_train":
    #     x, y = shuffle(x, y, r_seed=987)
    #     x = x[:num_train]
    #     y = y[:num_train]
    # else:
    #     x, y = shuffle(x, y, r_seed=987)
    #     x = x[num_train:]
    #     y = y[num_train:]

    if isshuffle:
        x, y = shuffle(x, y, r_seed=None)
    if num_load is not None:
        num_full_data = len(x)
        end = poiter + num_load
        if end > num_full_data:
            end = num_full_data
        x = x[poiter:end]
        y = y[poiter:end]

    return x, y


def load_mat_file_single_label(filename, min_load=10000000):
    filename_list = ['mnist', 'stl32', 'synsign', 'gtsrb', 'cifar32', 'usps32']
    data = loadmat(filename)
    x = data['X']
    y = data['y']
    num_load = min(min_load, len(x))

    if any(fn in filename for fn in filename_list):
        if 'mnist32_60_10' not in filename and 'mnistg' not in filename:
            y = y[0]
        else:
            y = np.argmax(y, axis=1)
    # process one-hot label encoder
    elif len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    x = x[:num_load]
    y = y[:num_load]
    print(filename, x.shape, y.shape)
    return x, y

# def scalar2onehot(scalars, num_class):
#     out_shape = scalars.shape + tuple([num_class])
#     idx_class = scalars.flatten()
#     n_data = len(idx_class)
#     onehots = np.zeros([n_data, num_class])
#     onehots[np.arange(n_data), idx_class] = 1
#     return onehots.reshape(out_shape)


def u2t(x):
    """Convert uint8 to [-1, 1] float
    """
    max_num = 50000
    # max_cast = 255
    max_cast = np.max(x)
    if len(x) > max_num:
        y = np.empty_like(x, dtype='float32')
        for i in range(len(x) // max_num):
            y[i*max_num: (i+1)*max_num] = (x[i*max_num: (i+1)*max_num].astype('float32') / max_cast) * 2 - 1

        y[(i + 1) * max_num:] = (x[(i + 1) * max_num:].astype('float32') / max_cast) * 2 - 1
    else:
        y = (x.astype('float32') / max_cast) * 2 - 1
    return y


class DataLoader:
    def __init__(self, src_domain=['mnistm'], trg_domain=['mnist'], data_path='./dataset', data_format='mat',
                 shuffle_data=False, dataset_name='digits', cast_data=True, num_class=10):
        # src_domain = src_domain.split(',')
        trg_domain = trg_domain.split(',')

        self.num_src_domain = len(src_domain)
        self.src_domain_name = src_domain
        self.trg_domain_name = trg_domain
        self.data_path = data_path
        self.data_format = data_format
        self.shuffle_data = shuffle_data
        self.dataset_name = dataset_name_map[src_domain[0]][0]
        self.cast_data = cast_data

        self.num_class = num_class
        self.src_train = {}  # {src_idx: ['src_name', x_train, y_train]}
        self.trg_train = {}  # {trg_idx: ['trg_name', x_train, y_train]}
        self.src_test = {}
        self.trg_test = {}

        print("Source domains", self.src_domain_name)
        print("Target domain", self.trg_domain_name)
        print("----- Training data -----")
        self._load_data_train()
        print("----- Test data -----")
        self._load_data_test()

        self.data_shape = self.src_train[0][1][0].shape
        self.num_domain = len(self.src_train.keys())

    def _load_data_train(self, tail_name="_train"):
        if not self.src_train:
            self.src_train = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
            self.trg_train = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def _load_data_test(self, tail_name="_test"):
        if not self.src_test:
            self.src_test = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
            self.trg_test = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def _reload_data_train(self, tail_name="_train"):
        self.src_train = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
        self.trg_train = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def _reload_data_test(self, tail_name="_test"):
        self.src_test = self._load_file(self.src_domain_name, tail_name, self.shuffle_data)
        self.trg_test = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data)

    def reload_data_train(self, tail_name="_train"):
        self.src_train = self._load_file(self.src_domain_name, tail_name, self.shuffle_data, reload=True)
        self.trg_train = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data, reload=True)

    def reload_data_test(self, tail_name="_test"):

        self.src_test = self._load_file(self.src_domain_name, tail_name, self.shuffle_data, reload=True)
        self.trg_test = self._load_file(self.trg_domain_name, tail_name, self.shuffle_data, reload=True)

    def _load_file(self, name_file=[], tail_name="_train", shuffle_data=False, reload=False):
        data_list = {}

        for idx, s_n in enumerate(name_file):
            src_file_name = dataset_name_map[s_n][-1]
            file_path_train = os.path.join(self.data_path, '{}{}.{}'.format(src_file_name, tail_name, self.data_format))
            # file_path_train = os.path.join(self.data_path, '{}{}.{}'.format(s_n, "_train", self.data_format))
            # print(file_path_train)
            if os.path.isfile(file_path_train):
                if self.dataset_name == 'digits':
                    min_load = 10000000
                    # min_load = 25000 if tail_name == '_train' else 9000
                    x_train, y_train = load_mat_file_single_label(file_path_train, min_load)
                elif self.dataset_name == 'office_caltech10_DECAF_feat':
                    x_train, y_train = load_mat_office_caltech10_decaf(file_path_train)
                elif self.dataset_name == 'office_caltech10_ResNet101_feat':
                    x_train, y_train = load_mat_office_caltech10_ResNet101(file_path_train, tail_name)
                elif self.dataset_name == 'office31_resnet50_feature':
                    x_train, y_train = load_office31_resnet50_feature(file_path_train)
                elif self.dataset_name == 'domain_net_ResNet101_feat':
                    if tail_name == "_train":
                        num_load = 345 * 100000
                        x_train, y_train = load_mat_domain_net_ResNet101(file_path_train, num_load=num_load, isshuffle=True, tail=tail_name, num_class=self.num_class)
                        shuffle_data = True
                    else:
                        num_load = 345 * 1000
                        x_train, y_train = load_mat_domain_net_ResNet101(file_path_train, num_load=num_load, isshuffle=True, tail=tail_name, num_class=self.num_class)
                        shuffle_data = True
                if shuffle_data:
                    x_train, y_train = shuffle(x_train, y_train)

                if 'mnist32_60_10' not in s_n and self.cast_data:
                    x_train = u2t(x_train)
                data_list.update({idx: [s_n, x_train, to_categorical(y_train, num_classes=self.num_class)]})
                # for i in range(4):
                #     cap = "{}_{}".format(s_n, y_scalar[i])
                #     self.visual_img(x_train[i], cap)
            else:
                print(file_path_train)
                raise('File not found!')
            if not reload:
                print(s_n, x_train.shape[0], x_train.min(), x_train.max(), "Label", y_train.min(), y_train.max())
        return data_list

    def visualize_mini_batch_class_balanced(self, sample_size=5, shuffle=False, num_tests=400):
        ''' sample the mini-batch with class balanced
        '''
        assert num_tests % self.num_class == 0
        for k in range(self.num_domain):
            label = self.src_train[k][2]
            label = np.argmax(label, axis=1)
            n_class = len(np.unique(label))

            s_index_each_class = []  # shape (10, 40)
            for i in range(n_class):
                s_index = np.nonzero(label == i)
                s_index_each_class.append(s_index[0][:num_tests//self.num_class])

            s_index_lst = []
            while len(s_index_each_class[0]) != 0:
                for i in range(n_class):
                    s_index_lst.extend(s_index_each_class[i][:sample_size])
                    s_index_each_class[i] = np.delete(s_index_each_class[i], np.s_[:sample_size])

            s_index_lst = np.array(s_index_lst, dtype=int)
            self.src_train[k][1] = self.src_train[k][1][s_index_lst]
            self.src_train[k][2] = self.src_train[k][2][s_index_lst]


    def onehot2scalar(self, onehot_vectors, axis=1):
        return np.argmax(onehot_vectors, axis=axis)

    def load_batch_src(self, batch_size=64, mini_batch=None, domain_idx=None, y_specific=None, shuffle=True):
        '''
        :param batch_size: max number of data to load ( batch_size % len(domain_idx) = 0)
        :param mini_batch: mini_batch for each domain in domain_idx, if None -> minichatch = batch/num_domain
        :param domain_idx: idx domain to load data, idx < 0: target domain at 1 - idx
        :param y_specific: label of data to load
        :return:
        '''

        return x_sample, y_onehot_sample, y_scalar_sample, y_d_onehot_sample

    def load_batch_trg(self, batch_size=64, shuffle=True):
        '''

        :param batch_size: max number of data to load ( batch_size % len(domain_idx) = 0)
        :param mini_batch: mini_batch for each domain in domain_idx, if None -> minichatch = batch/num_domain
        :param domain_idx: idx domain to load data, idx < 0: target domain at 1 - idx
        :param y_specific: label of data to load
        :return:
        '''

        self._load_data_train()

        x_idx, y_1hot_idx, y_sca_idx = self.trg_train[0]

        idx_y_spe = np.arange(len(y_sca_idx))

        if shuffle:
            np.random.shuffle(idx_y_spe)
        x_sample = x_idx[idx_y_spe[:batch_size]]
        # y_onehot_sample = y_1hot_idx[idx_y_spe][:batch_size]
        # y_scalar_sample = y_sca_idx[idx_y_spe][:batch_size]

        return x_sample, [], [], []

    def load_batch_test_src(self, batch_size=64, mini_batch=None, domain_idx=None, y_specific=-1, shuffle=True):
        '''

        :param batch_size: max number of data to load ( batch_size % len(domain_idx) = 0)
        :param mini_batch: mini_batch for each domain in domain_idx, if None -> minichatch = batch/num_domain
        :param domain_idx: idx domain to load data, idx < 0: target domain at 1 - idx
        :param y_specific: label of data to load, -1 mean all label
        :return:
        '''

        self._load_data_test()
        if domain_idx is None:
            domain_idx = np.arange(self.num_domain)

        x_sample = []
        y_onehot_sample = []
        y_scalar_sample = []
        y_d_onehot_sample = []
        for idx in domain_idx:
            x_idx, y_1hot_idx, y_sca_idx = self.src_test[idx]

            if y_specific == -1:
                idx_y_spe = np.arange(len(y_sca_idx))
            else:
                idx_y_spe = np.argwhere(y_sca_idx == y_specific).reshape(-1)

            if shuffle:
                np.random.shuffle(idx_y_spe)

            if batch_size is not None:
                mini_batch = batch_size // len(domain_idx)
                idx_y_spe = idx_y_spe[:mini_batch]

            y_d_load = np.zeros([len(idx_y_spe), self.num_domain])
            y_d_load[:, idx] = 1

            x_load = x_idx[idx_y_spe]
            y_1hot_load = y_1hot_idx[idx_y_spe]
            y_sca_load = y_sca_idx[idx_y_spe]

            x_sample.append(x_load)
            y_onehot_sample.append(y_1hot_load)
            y_scalar_sample.append(y_sca_load)
            y_d_onehot_sample.append(y_d_load)

        return x_sample, y_onehot_sample, y_scalar_sample, y_d_onehot_sample

    def load_batch_test_trg(self, batch_size=None, shuffle=True, y_specific=-1):
        '''
        :param batch_size: max number of data to load ( batch_size % len(domain_idx) = 0)
        :return:
        '''

        self._load_data_test()

        x_idx, y_1hot_idx, y_sca_idx = self.trg_test[0]
        if y_specific == -1:
            idx_y_spe = np.arange(len(y_sca_idx))
        else:
            idx_y_spe = np.argwhere(y_sca_idx == y_specific).reshape(-1)

        if shuffle:
            np.random.shuffle(idx_y_spe)
        if batch_size is not None:
            idx_y_spe = idx_y_spe[:batch_size]

        x_sample = x_idx[idx_y_spe]
        y_onehot_sample = y_1hot_idx[idx_y_spe]
        y_scalar_sample = y_sca_idx[idx_y_spe]

        return x_sample, y_onehot_sample, y_scalar_sample, []

    def visual_img(self, img, caption="", save=False, save_path="log_model"):
        img += 1
        img /= 2
        if save:
            img *= 255
            s_p = os.path.join(save_path, caption + ".png")
            cv2.imwrite(s_p, img)
        else:
            cv2.imshow(caption, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # digit = ["mnist", "mnistm", "svhn", "syn"]
    digit = ["usps"]
    ld = DataLoader(src_domain=digit, trg_domain=["syn"], data_path="./dataset/digit-datasets")
    # x_sample, y_onehot_sample, y_scalar_sample, y_d_onehot_sample = ld.load_batch_src(mini_batch=7, y_specific=-1,
    #                                                                              domain_idx=[-1, 0, 1])
    #     # for i in range(ld.num_domain + 1):
    #     img = x_sample[i]
    #     for j in range(4):
    #         s = "label={}\tl_one=\t{}\tl_d=\{}".format(y_scalar_sample[i][j], y_onehot_sample[i][j], y_d_onehot_sample[i][j])
    #         ld.visual_img(img[j], caption=s)

    x_sample, y_onehot_sample, y_scalar_sample, y_d_onehot_sample = ld.load_batch_test_trg(batch_size=10)

    for j in range(4):
        s = "label={}_l_one={}".format(y_scalar_sample[j], y_onehot_sample[j])
        ld.visual_img(x_sample[j], caption=s, save=True)

    data_name = "clipart_ResNet101,infograph_ResNet101,painting_ResNet101,quickdraw_ResNet101,real_ResNet101,sketch_ResNet101"
