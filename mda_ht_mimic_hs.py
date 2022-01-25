from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope
from tensorbayes.layers import dense, conv2d, batch_norm, instance_norm
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits as sigmoid_x_entropy
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_x_entropy_two

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from generic_utils import random_seed

from layers import leaky_relu
import os
from generic_utils import model_dir
# import scipy.misc
# import time
# import datetime
import numpy as np
from sklearn import metrics
import tensorbayes as tb
from layers import batch_ema_acc, batch_teac_stud_avg_acc, batch_teac_stud_ent_acc
# import random
# from matplotlib import pyplot
# import pandas as pd
from collections import OrderedDict
# from scipy.spatial.distance import cdist
from keras.utils.np_utils import to_categorical
from collections import Counter
import json


def build_block(input_layer, layout, info=1):
    x = input_layer
    for i in range(0, len(layout)):
        with tf.variable_scope('l{:d}'.format(i)):
            f, f_args, f_kwargs = layout[i]
            x = f(x, *f_args, **f_kwargs)
            if info > 1:
                print(x)
    return x


@add_arg_scope
def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=np.arange(1, len(d.shape)))
    return output


def build_encode_template(
        input_layer, training_phase, scope, encode_layout,
        reuse=None, internal_update=False, getter=None, inorm=True, data_name='large'):
    # if inorm is 1, we normalize the data otherwise we do not normalize it
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            preprocess = instance_norm if inorm else tf.identity

            layout = encode_layout(preprocess=preprocess, training_phase=training_phase, data_name=data_name)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_class_discriminator_template(
        input_layer, training_phase, scope, num_classes, class_discriminator_layout,
        reuse=None, internal_update=False, getter=None, data_name='digitFive'):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=training_phase), \
             arg_scope([batch_norm], internal_update=internal_update):
            layout = class_discriminator_layout(num_classes=num_classes, global_pool=True, activation=None,
                                                data_name=data_name)
            output_layer = build_block(input_layer, layout)

    return output_layer


def build_domain_discriminator_template(x, domain_layout, c=1, reuse=None, scope='domain_disc', data_name='digitFive'):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            layout = domain_layout(c=c, data_name=data_name)
            output_layer = build_block(x, layout)

    return output_layer


def build_disc_src_trg_network_template(x, domain_layout, c=1, reuse=None, data_name='digitFive'):
    with tf.variable_scope('disc_src_trg_net', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            layout = domain_layout(c=c, data_name=data_name)
            output_layer = build_block(x, layout)

    return output_layer


def get_default_config():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    return tf_config


class STEM():
    def __init__(self,
                 model_name="STEM-results",
                 learning_rate=0.001,
                 batch_size=128,
                 num_iters=80000,
                 phase1_iters=20000,
                 summary_freq=400,
                 src_class_trade_off=1.0,
                 src_domain_trade_off='1.0,1.0',
                 # src_vat_trade_off=1.0,
                 trg_trade_off=0.001,
                 trg_ent_trade_off=0.001,
                 gen_trade_off=0.1,
                 domain_trade_off=0.1,
                 mimic_troff=0.1,
                 encode_layout=None,
                 classify_layout=None,
                 domain_layout=None,
                 disc_layout=None,
                 current_time='',
                 inorm=True,
                 mdaot_model_id='',
                 save_grads=False,
                 only_save_final_model=True,
                 data_name='digitFive',
                 data_augmentation=False,
                 sample_size=50,
                 trg_ent_src_domain_trdoff=0.1,
                 # assign_ith_hs=0,
                 process='origin',
                 **kwargs):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.phase1_iters = phase1_iters
        self.summary_freq = summary_freq
        self.src_class_trade_off = src_class_trade_off
        self.src_domain_trade_off = [float(item) for item in src_domain_trade_off.split(',')]
        # self.src_vat_trade_off = src_vat_trade_off
        self.trg_trade_off = trg_trade_off
        self.trg_ent_trade_off = trg_ent_trade_off
        self.gen_trade_off = gen_trade_off
        self.domain_trade_off = domain_trade_off
        self.mimic_troff = mimic_troff

        self.encode_layout = encode_layout
        self.classify_layout = classify_layout
        self.domain_layout = domain_layout
        self.disc_layout = disc_layout

        self.current_time = current_time
        self.inorm = inorm

        self.mdaot_model_id = mdaot_model_id

        self.save_grads = save_grads
        self.only_save_final_model = only_save_final_model

        self.data_name = data_name

        self.sample_size = sample_size
        self.trg_ent_src_domain_trdoff = trg_ent_src_domain_trdoff
        # self.assign_ith_hs = assign_ith_hs

        self.data_augmentation = data_augmentation
        self.image_datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            horizontal_flip=True,
            vertical_flip=False)
        self.process = process

        # only for CIFAR-STL
        # self.image_datagen = ImageDataGenerator(
        #     rotation_range=0,
        #     width_shift_range=0.15,
        #     height_shift_range=0.15,
        #     shear_range=0.,
        #     zoom_range=0.,
        #     channel_shift_range=0.,
        #     horizontal_flip=True,
        #     vertical_flip=False,
        #     rescale=1./255
        # )

    def _init(self, data_loader):
        np.random.seed(random_seed())
        tf.set_random_seed(random_seed())
        tf.reset_default_graph()

        self.tf_graph = tf.get_default_graph()
        self.tf_config = get_default_config()
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        self.data_loader = data_loader
        # self.src_num_classes = len(np.unique(y_src_train))
        # self.trg_num_classes = len(np.unique(y_trg_train))
        self.num_classes = self.data_loader.num_class
        self.batch_size_src = self.sample_size*self.num_classes

        assert len(self.src_domain_trade_off) == self.data_loader.num_src_domain
        assert self.sample_size*self.num_classes*self.data_loader.num_src_domain == self.batch_size

    def _get_variables(self, list_scopes):
        variables = []
        for scope_name in list_scopes:
            variables.append(tf.get_collection('trainable_variables', scope_name))
        return variables

    def convert_one_hot(self, y):
        y_scalar = y.copy()
        y_idx = y_scalar.reshape(-1).astype(int) if y_scalar is not None else None
        y_scalar = np.eye(self.num_classes)[y_idx] if y_scalar is not None else None
        return y_scalar

    def _get_scope(self, part_name, side_name, same_network=True):
        suffix = ''
        if not same_network:
            suffix = '/' + side_name
        return part_name + suffix

    def _get_teacher_scopes(self):
        return ['generator', 'classifier', 'domain_disc']

    def _get_student_primary_scopes(self):
        return ['generator', 'c-trg']

    def _get_student_secondary_scopes(self):
        return ['disc_src_trg_net']

    def _build_source_middle(self, x_src, is_reused):
        scope_name = self._get_scope('generator', 'src')
        if is_reused == 0:
            generator_model = build_encode_template(x_src, encode_layout=self.encode_layout,
                                                    scope=scope_name,
                                                    training_phase=self.is_training,
                                                    inorm=self.inorm,
                                                    internal_update=True,
                                                    data_name=self.data_name)
        else:
            generator_model = build_encode_template(x_src, encode_layout=self.encode_layout,
                                                    scope=scope_name,
                                                    training_phase=self.is_training,
                                                    inorm=self.inorm,
                                                    # original
                                                    reuse=True,
                                                    internal_update=True,
                                                    # reuse=True, internal_update=False,
                                                    data_name=self.data_name)
        return generator_model

    def _build_target_middle(self, x_trg, reuse=None):
        # scope_name = self._get_scope('generator', 'trg', same_network=False)
        scope_name = 'generator'
        return build_encode_template(
            x_trg, encode_layout=self.encode_layout,
            scope=scope_name, training_phase=self.is_training, inorm=self.inorm,
            reuse=reuse, internal_update=True, data_name=self.data_name
        )  # reuse the 'encode_layout'

    def _build_classifier(self, x, num_classes, ema=None, is_teacher=False):
        g_teacher_scope = self._get_scope('generator', 'teacher', same_network=False)
        g_x = build_encode_template(
            x, encode_layout=self.encode_layout,
            scope=g_teacher_scope if is_teacher else 'generator', training_phase=False, inorm=self.inorm,
            reuse=False if is_teacher else True, getter=None if is_teacher else tb.tfutils.get_getter(ema),
            data_name=self.data_name
        )

        h_teacher_scope = self._get_scope('c-trg', 'teacher', same_network=False)
        h_g_x = build_class_discriminator_template(
            g_x, training_phase=False, scope=h_teacher_scope if is_teacher else 'c-trg', num_classes=num_classes,
            reuse=False if is_teacher else True, class_discriminator_layout=self.classify_layout,
            getter=None if is_teacher else tb.tfutils.get_getter(ema), data_name=self.data_name
        )
        return h_g_x  # have not been applied softmax

    def _build_domain_discriminator(self, x_mid, reuse=None, scope='domain_disc'):
        return build_domain_discriminator_template(x_mid, domain_layout=self.domain_layout, c=self.data_loader.num_src_domain, reuse=reuse, scope=scope, data_name=self.data_name)

    def _build_disc_src_trg_network(self, x_mid, reuse=None):
        return build_disc_src_trg_network_template(x_mid, domain_layout=self.disc_layout, c=1, reuse=reuse, data_name=self.data_name)

    def _build_class_src_discriminator(self, x_src, num_src_classes, i, reuse=None):
        # if is_reused == 0:
        #     classifier_model = build_class_discriminator_template(
        #     x_src, training_phase=self.is_training, scope='classifier', num_classes=num_src_classes,
        #     class_discriminator_layout=self.classify_layout, data_name=self.data_name
        #     )
        # else:
        #     classifier_model = build_class_discriminator_template(
        #         x_src, training_phase=self.is_training, scope='classifier', num_classes=num_src_classes,
        #         reuse=True, internal_update=True, class_discriminator_layout=self.classify_layout, data_name=self.data_name
        #     )

        classifier_model = build_class_discriminator_template(
            x_src, training_phase=self.is_training, scope='classifier/{}'.format(i), num_classes=num_src_classes,
            reuse=reuse, internal_update=True, class_discriminator_layout=self.classify_layout, data_name=self.data_name
        )
        return classifier_model

    def _build_class_trg_discriminator(self, x_trg, num_trg_classes):
        return build_class_discriminator_template(
            x_trg, training_phase=self.is_training, scope='c-trg', num_classes=num_trg_classes,
            reuse=False, internal_update=True, class_discriminator_layout=self.classify_layout, data_name=self.data_name
        )

    def perturb_image(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                      pert='vat', scope=None, radius=3.5, scope_classify=None, scope_encode=None, training_phase=None):
        with tf.name_scope(scope, 'perturb_image'):
            eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

            # Predict on randomly perturbed image
            x_eps_mid = build_encode_template(
                x + eps, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, reuse=True,
                inorm=self.inorm, data_name=self.data_name)
            x_eps_pred = build_class_discriminator_template(
                x_eps_mid, class_discriminator_layout=class_discriminator_layout,
                training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                data_name=self.data_name
            )
            # eps_p = classifier(x + eps, phase=True, reuse=True)
            loss = softmax_x_entropy_two(labels=p, logits=x_eps_pred)

            # Based on perturbed image, get direction of greatest error
            eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

            # Use that direction as adversarial perturbation
            eps_adv = normalize_perturbation(eps_adv)
            x_adv = tf.stop_gradient(x + radius * eps_adv)

        return x_adv

    def vat_loss(self, x, p, num_classes, class_discriminator_layout, encode_layout,
                 scope=None, scope_classify=None, scope_encode=None, training_phase=None):

        with tf.name_scope(scope, 'smoothing_loss'):
            x_adv = self.perturb_image(
                x, p, num_classes, class_discriminator_layout=class_discriminator_layout, encode_layout=encode_layout,
                scope_classify=scope_classify, scope_encode=scope_encode, training_phase=training_phase)

            x_adv_mid = build_encode_template(
                x_adv, encode_layout=encode_layout, scope=scope_encode, training_phase=training_phase, inorm=self.inorm,
                reuse=True, data_name=self.data_name)
            x_adv_pred = build_class_discriminator_template(
                x_adv_mid, training_phase=training_phase, scope=scope_classify, reuse=True, num_classes=num_classes,
                class_discriminator_layout=class_discriminator_layout, data_name=self.data_name
            )
            # p_adv = classifier(x_adv, phase=True, reuse=True)
            loss = tf.reduce_mean(softmax_x_entropy_two(labels=tf.stop_gradient(p), logits=x_adv_pred))

        return loss

    def _build_vat_loss(self, x, p, num_classes, scope=None, scope_classify=None, scope_encode=None):
        return self.vat_loss(  # compute the divergence between C(x) and C(G(x+r))
            x, p, num_classes,
            class_discriminator_layout=self.classify_layout,
            encode_layout=self.encode_layout,
            scope=scope, scope_classify=scope_classify, scope_encode=scope_encode,
            training_phase=self.is_training
        )

    def _compute_cosine_similarity(self, x_trg_mid, x_src_mid_all):  # x_src_mid= x_trg_mid
        x_trg_mid_flatten = tf.layers.Flatten()(x_trg_mid)
        x_src_mid_all_flatten = tf.layers.Flatten()(x_src_mid_all)
        # Cosine similarity
        similarity = tf.reduce_sum(x_trg_mid_flatten[:, tf.newaxis] * x_src_mid_all_flatten, axis=-1)
        # Only necessary if vectors are not normalized
        similarity /= tf.norm(x_trg_mid_flatten[:, tf.newaxis], axis=-1) * tf.norm(x_src_mid_all_flatten, axis=-1)
        # If you prefer the distance measure
        distance = 1.0 - similarity
        return distance

    def _compute_teacher_hs(self, y_label_trg_output_each_h, y_d_trg_sofmax_output):
        # (mini_batch, num_domain)
        # y_d_trg_sofmax_output = y_src_discriminator_prob

        # input_h_domain = [latent_v_trg for _ in range(num_domain)]  # (num_domain, batch_size, latent_dim)
        # y_label_trg_output_each_h = y_src_logit_all  # (num_domain, batch_size, num_class)

        # num_domain, (mini_batch, num_class) -> (mini_batch, num_domain, num_class)
        y_label_trg_output_each_h = tf.transpose(tf.stack(y_label_trg_output_each_h), perm=[1, 0, 2])
        # y_label_trg_output_each_h = tf.nn.softmax(y_label_trg_output_each_h)  # true code
        # y_label_trg_output_each_h = y_label_trg_output_each_h

        # compute c for each y
        y_d_trg_sofmax_output_multi_y = y_d_trg_sofmax_output

        # change dim y_d_trg_sofmax_output_change to (mini_batch, num_domain, num_class)
        y_d_trg_sofmax_output_multi_y = tf.expand_dims(y_d_trg_sofmax_output_multi_y, axis=-1)
        y_d_trg_sofmax_output_multi_y = tf.tile(y_d_trg_sofmax_output_multi_y, [1, 1, self.num_classes])

        # (mini_batch,num_domain, num_class)
        y_label_trg_output = y_d_trg_sofmax_output_multi_y * y_label_trg_output_each_h

        # (mini_batch, num_class)
        y_label_trg_output = tf.reduce_sum(y_label_trg_output, axis=1)
        # y_label_trg_softmax_output = tf.nn.softmax(y_label_trg_output)
        return y_label_trg_output

    def get_distances(self, a, b, name='L2'):
        if name == 'L1':
            return tf.reduce_sum(tf.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)
        elif name == 'L2':
            return tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))

    def _build_model(self):
        # x_src, x_trg (batch_size, width, height, number of channel)
        self.x_src_lst = []
        self.y_src_lst = []
        for i in range(self.data_loader.num_src_domain):
            x_src = tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_src,
                                        name='x_src_{}_input'.format(i))  # name='x_src_input'
            y_src = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes),
                                        name='y_src_{}_input'.format(i))

            self.x_src_lst.append(x_src)
            self.y_src_lst.append(y_src)

        self.x_trg = tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_trg, name='x_trg_input')
        self.y_trg = tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes),
                                    name='y_trg_input')  # (batch_size, num_classes)
        self.y_src_domain = tf.placeholder(dtype=tf.float32, shape=(None, self.data_loader.num_src_domain),
                                    name='y_src_domain_input')

        T = tb.utils.TensorDict(dict(
            x_tmp=tf.placeholder(dtype=tf.float32, shape=tuple([None]) + self.dim_src),
            y_tmp=tf.placeholder(dtype=tf.float32, shape=(None, self.num_classes))
        ))

        self.is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

        # latent_v_src_lst
        self.x_src_mid_lst = []
        for i in range(self.data_loader.num_src_domain):
            x_src_mid = self._build_source_middle(self.x_src_lst[i], is_reused=i)  # (bs/k, 8, 8, 64)
            self.x_src_mid_lst.append(x_src_mid)
        # latent_v_trg
        self.x_trg_mid = self._build_target_middle(self.x_trg, reuse=True)

        # <editor-fold desc="Classifier-logits">
        self.y_src_logit_lst = []
        for i in range(self.data_loader.num_src_domain):
            # y_src_logit = self._build_class_src_discriminator(self.x_src_mid_lst[i], self.num_classes,
            #                                                        is_reused=i)  # (batch_size, src_num_classes)
            y_src_logit = self._build_class_src_discriminator(self.x_src_mid_lst[i], self.num_classes, i)  # (bs/k, n_classes)
            self.y_src_logit_lst.append(y_src_logit)

        self.y_trg_logit = self._build_class_trg_discriminator(self.x_trg_mid,
                                                               self.num_classes)  # (batch_size, trg_num_classes)
        # </editor-fold>

        # <editor-fold desc="Classification">
        self.src_loss_class_lst = []
        self.src_loss_class_sum = tf.constant(0.0)  # will be added to total loss
        for i in range(self.data_loader.num_src_domain):
            # <editor-fold desc="label smoothing for hs_i(G(xs)">
            # self.y_src_i_smooth = (1 - 0.1) * self.y_src_lst[i] + 0.1 / self.num_classes
            # src_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=self.y_src_logit_lst[i], labels=self.y_src_i_smooth)  # (batch_size,)
            # </editor-fold>
            # original
            src_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.y_src_logit_lst[i], labels=self.y_src_lst[i])  # (batch_size,)
            src_loss_class = tf.reduce_mean(src_loss_class_detail)  # real number
            self.src_loss_class_lst.append(self.src_domain_trade_off[i]*src_loss_class)
            self.src_loss_class_sum += self.src_domain_trade_off[i]*src_loss_class

        self.trg_loss_class_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_trg_logit, labels=self.y_trg)
        self.trg_loss_class = tf.reduce_mean(self.trg_loss_class_detail)  # just use for testing
        # </editor-fold>

        # <editor-fold desc="Source domain discriminator - C">
        self.x_src_mid_all = tf.concat(self.x_src_mid_lst, axis=0)  # (bs, 8, 8, 64)
        self.y_src_discriminator_logit = self._build_domain_discriminator(self.x_src_mid_all)  # (bs, k)

        # <editor-fold desc="label smoothing for C(G(xs))">
        # self.y_src_domain_smooth = (1 - 0.1) * self.y_src_domain + 0.1 / self.data_loader.num_src_domain
        # self.src_loss_discriminator_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
        #     logits=self.y_src_discriminator_logit, labels=self.y_src_domain_smooth)  # (bs,)
        # </editor-fold>
        # original
        self.src_loss_discriminator_detail = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_src_discriminator_logit, labels=self.y_src_domain)  # (bs,)
        self.src_loss_discriminator = tf.reduce_mean(self.src_loss_discriminator_detail)  # will be added to total loss
        # </editor-fold>

        # <editor-fold desc="Compute teacher hS(xS)">
        self.y_src_teacher_all = []
        for i, bs in zip(range(self.data_loader.num_src_domain),
                         range(0, self.batch_size_src * self.data_loader.num_src_domain, self.batch_size_src)):
            y_src_logit_each_h_lst = []
            for j in range(self.data_loader.num_src_domain):
                y_src_logit_each_h = self._build_class_src_discriminator(self.x_src_mid_lst[i], self.num_classes,
                                                                  j, reuse=True)
                y_src_logit_each_h_lst.append(y_src_logit_each_h)
            y_src_logit_each_h_lst = tf.nn.softmax(tf.convert_to_tensor(y_src_logit_each_h_lst))  # (k, bs/k, n_classes)
            # slice the output of src discriminator to compute corresponding hS_i
            y_src_discriminator_prob = tf.nn.softmax(tf.gather(self.y_src_discriminator_logit,
                                                               tf.range(bs, bs + self.batch_size_src,
                                                                        dtype=tf.int32), axis=0))  # (bs/k, k)
            y_src_teacher = self._compute_teacher_hs(y_src_logit_each_h_lst, y_src_discriminator_prob)  # (bs/k, n_classes)
            self.y_src_teacher_all.append(y_src_teacher)
        self.y_src_teacher_all = tf.concat(self.y_src_teacher_all, axis=0)  # (bs, n_classes)

        # <editor-fold desc="Compute teacher hS(G(xt)), just use for observation">
        y_trg_logit_each_h_lst = []
        for j in range(self.data_loader.num_src_domain):
            y_trg_logit_each_h = self._build_class_src_discriminator(self.x_trg_mid, self.num_classes,
                                                                     j, reuse=True)
            y_trg_logit_each_h_lst.append(y_trg_logit_each_h)
        y_trg_logit_each_h_lst = tf.nn.softmax(tf.convert_to_tensor(y_trg_logit_each_h_lst))  # (k, bs, n_classes)
        self.y_trg_src_domains_logit = self._build_domain_discriminator(self.x_trg_mid, reuse=True)  # (bs, k)
        y_trg_discriminator_prob = tf.nn.softmax(self.y_trg_src_domains_logit)  # (bs, k)
        self.y_trg_teacher = self._compute_teacher_hs(y_trg_logit_each_h_lst, y_trg_discriminator_prob)  # (bs, n_classes)
        # </editor-fold>

        # <editor-fold desc="Compute l2 distance between centroid(batch_src) and centroid(batch_trg)">
        # self.distance_2_centroids = []
        # x_src_mid_all_flt = tf.layers.Flatten()(self.x_src_mid_all)
        # x_trg_mid_flt = tf.layers.Flatten()(self.x_trg_mid)
        # centroid_all_x_src = tf.reduce_mean(x_src_mid_all_flt, axis=0, keepdims=True)
        # centroid_all_x_trg = tf.reduce_mean(x_trg_mid_flt, axis=0, keepdims=True)
        # for i in range(self.data_loader.num_src_domain):
        #     x_src_mid_i_flt = tf.layers.Flatten()(self.x_src_mid_lst[i])
        #     centroid_x_src_i = tf.reduce_mean(x_src_mid_i_flt, axis=0, keepdims=True)
        #     self.distance_2_centroids.append(tf.norm(centroid_x_src_i - centroid_all_x_trg, ord='euclidean'))
        #
        # self.distance_2_centroids.append(tf.norm(centroid_all_x_src - centroid_all_x_trg, ord='euclidean'))
        # </editor-fold>

        # <editor-fold desc="mimic loss">
        self.ht_g_xs = build_class_discriminator_template(
            self.x_src_mid_all, training_phase=self.is_training, scope='c-trg', num_classes=self.num_classes,
            # original
            reuse=True, internal_update=True, class_discriminator_layout=self.classify_layout, data_name=self.data_name
            # reuse=True, internal_update=False, class_discriminator_layout=self.classify_layout, data_name=self.data_name
        )
        # original
        self.mimic_loss_ht_g_xs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.ht_g_xs, labels=self.y_src_teacher_all))

        self.mimic_loss_ht_g_xt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.y_trg_logit, labels=self.y_trg_teacher))

        # original
        self.mimic_loss_va = self.mimic_loss_ht_g_xs + self.mimic_loss_ht_g_xt

        self.trg_loss_vat = self._build_vat_loss(
            self.x_trg, self.y_trg_logit, self.num_classes,
            # because we need to call encode_template (generator), so we need to have scope_encode
            # and we need to call classifier, so need need to have scope_classify
            scope_encode=self._get_scope('generator', 'trg'), scope_classify='c-trg'
        )
        # </editor-fold>

        # <editor-fold desc="GAN loss">
        self.x_fr_src = self._build_disc_src_trg_network(self.x_src_mid_all)
        self.x_fr_trg = self._build_disc_src_trg_network(self.x_trg_mid, reuse=True)

        # # TUAN EDIT adds term to control gan loss - haven't debug yet
        # self.trg_loss_entropy_src_domains = softmax_x_entropy_two(labels=self.y_trg_src_domains_logit,
        #                                                                          logits=self.y_trg_src_domains_logit)
        # self.trg_troff_gan = 1.0 + tf.exp(-self.trg_loss_entropy_src_domains)
        # self.src_loss_entropy_src_domains = softmax_x_entropy_two(labels=self.y_src_discriminator_logit,
        #                                                           logits=self.y_src_discriminator_logit)
        # self.src_troff_gan = 1.0 + tf.exp(-self.src_loss_entropy_src_domains)
        # self.loss_gen = tf.reduce_mean(
        #     self.src_troff_gan * sigmoid_x_entropy(labels=tf.zeros_like(self.x_fr_src), logits=self.x_fr_src) +
        #     self.trg_troff_gan * sigmoid_x_entropy(labels=tf.ones_like(self.x_fr_trg), logits=self.x_fr_trg))
        # # discriminator loss (similar to discriminator loss in GAN)
        # self.loss_disc = tf.reduce_mean(
        #     self.src_troff_gan * sigmoid_x_entropy(labels=tf.ones_like(self.x_fr_src), logits=self.x_fr_src) +
        #     self.trg_troff_gan * sigmoid_x_entropy(labels=tf.zeros_like(self.x_fr_trg), logits=self.x_fr_trg))
        # TUAN EDIT adds term to control gan loss

        # original
        self.loss_gen = 0.5 * tf.reduce_mean(
            sigmoid_x_entropy(labels=tf.zeros_like(self.x_fr_src), logits=self.x_fr_src) +
            sigmoid_x_entropy(labels=tf.ones_like(self.x_fr_trg), logits=self.x_fr_trg))

        # discriminator loss (similar to discriminator loss in GAN)
        self.loss_disc = 0.5 * tf.reduce_mean(
            sigmoid_x_entropy(labels=tf.ones_like(self.x_fr_src), logits=self.x_fr_src) +
            sigmoid_x_entropy(labels=tf.zeros_like(self.x_fr_trg), logits=self.x_fr_trg))
        # </editor-fold>

        # <editor-fold desc="Conditional entropy loss w.r.t target distribution">
        self.trg_loss_cond_entropy = tf.reduce_mean(softmax_x_entropy_two(labels=self.y_trg_logit,
                                                                   logits=self.y_trg_logit))

        # <editor-fold desc="Conditional entropy loss w.r.t C(G(xt))">
        self.y_trg_src_domains_prob = tf.nn.softmax(self.y_trg_src_domains_logit)

        self.trg_loss_entropy_src_domains = tf.reduce_mean(softmax_x_entropy_two(labels=self.y_trg_src_domains_logit,
                                                                          logits=self.y_trg_src_domains_logit))
        # </editor-fold>

        # <editor-fold desc="Accuracy">
        # compute acc for each source domain
        self.src_accuracy_lst = []
        self.y_src_softmax_lst = []
        for i in range(self.data_loader.num_src_domain):
            y_src_pred = tf.argmax(self.y_src_logit_lst[i], 1, output_type=tf.int32)
            self.y_src_softmax_lst.append(y_src_pred)
            y_src_sparse = tf.argmax(self.y_src_lst[i], 1, output_type=tf.int32)
            src_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_src_sparse, y_src_pred), 'float32'))
            self.src_accuracy_lst.append(src_accuracy)
        # compute acc for target domain
        self.y_trg_pred = tf.argmax(self.y_trg_logit, 1, output_type=tf.int32)
        self.y_trg_sparse = tf.argmax(self.y_trg, 1, output_type=tf.int32)
        self.trg_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_trg_sparse, self.y_trg_pred), 'float32'))
        # compute acc for src domain disc
        self.y_src_domain_pred = tf.argmax(self.y_src_discriminator_logit, 1, output_type=tf.int32)
        self.y_src_domain_sparse = tf.argmax(self.y_src_domain, 1, output_type=tf.int32)
        self.src_domain_acc = tf.reduce_mean(tf.cast(tf.equal(self.y_src_domain_sparse, self.y_src_domain_pred), 'float32'))
        # compute acc for the teacher(xs), a combination of source classifiers
        self.y_src_all_sparse = tf.argmax(tf.concat(self.y_src_lst, axis=0), 1, output_type=tf.int32)  # (bs,)
        self.y_src_teacher_sparse = tf.argmax(self.y_src_teacher_all, 1, output_type=tf.int32)  # (bs,)
        self.src_teacher_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_src_teacher_sparse, self.y_src_all_sparse), 'float32'))
        # compute acc for teacher(xt)
        self.y_trg_teacher_sparse = tf.argmax(self.y_trg_teacher, 1, output_type=tf.int32)  # (bs,)
        self.trg_teacher_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_trg_teacher_sparse, self.y_trg_sparse), 'float32'))
        # </editor-fold>

        # Primary loss
        lst_phase1_losses = [
            (self.src_class_trade_off, self.src_loss_class_sum),
            # (self.gen_trade_off, self.OT_loss),
            (self.domain_trade_off, self.src_loss_discriminator),
            # (self.src_vat_trade_off, self.src_loss_vat),
            # (self.trg_trade_off, self.trg_loss_vat),
            # (self.trg_trade_off, self.trg_loss_cond_entropy),
            # (self.trg_ent_src_domain_trdoff, self.trg_loss_entropy_src_domains)
        ]
        self.phase1_loss = tf.constant(0.0)
        for trade_off, loss in lst_phase1_losses:
            # if trade_off != 0:
            self.phase1_loss += trade_off * loss

        lst_phase2_losses = [
            (self.src_class_trade_off, self.src_loss_class_sum),    # sum Hi(G(xs_i))
            (self.gen_trade_off, self.loss_gen),                    # loss_gan
            (self.domain_trade_off, self.src_loss_discriminator),   # C(G(xs))
            # (self.src_vat_trade_off, self.src_loss_vat),
            (self.trg_trade_off, self.trg_loss_vat),                # vat for Ht(G(xt))
            (self.trg_ent_trade_off, self.trg_loss_cond_entropy),       # entropy Ht(G(xt))
            (self.trg_ent_src_domain_trdoff, self.trg_loss_entropy_src_domains),  # entropy C(G(xt))
            (self.mimic_troff, self.mimic_loss_ht_g_xs)                     # Ht(G(xs))
        ]
        self.phase2_loss = tf.constant(0.0)
        for trade_off, loss in lst_phase2_losses:
            # if trade_off != 0:
            self.phase2_loss += trade_off * loss

        lst_phase2_losses_global = [
            (self.src_class_trade_off, self.src_loss_class_sum),    # sum Hi(G(xs_i))
            (self.gen_trade_off, self.loss_gen),                    # loss_gan
            (self.domain_trade_off, self.src_loss_discriminator),   # C(G(xs))
            # (self.src_vat_trade_off, self.src_loss_vat),
            (self.trg_trade_off, self.trg_loss_vat),                # vat for Ht(G(xt))
            (self.trg_ent_trade_off, self.trg_loss_cond_entropy),       # entropy Ht(G(xt))
            (self.trg_ent_src_domain_trdoff, self.trg_loss_entropy_src_domains),  # entropy C(G(xt))
            (self.mimic_troff, self.mimic_loss_va)                  # Ht(G(xs)) + Ht(G(xt)) -> label is hs(g(xt))
        ]
        print("src_class_trade_off", self.src_class_trade_off)
        print("gen_trade_off", self.gen_trade_off)
        print("domain_trade_off", self.domain_trade_off)
        print("trg_trade_off", self.trg_trade_off)
        print("trg_ent_trade_off", self.trg_ent_trade_off)
        print("trg_ent_src_domain_trdoff", self.trg_ent_src_domain_trdoff)
        print("mimic_troff", self.mimic_troff)

        self.phase2_loss_global = tf.constant(0.0)
        for trade_off, loss in lst_phase2_losses_global:
            # if trade_off != 0:
            self.phase2_loss_global += trade_off * loss

        primary_student_variables = self._get_variables(self._get_student_primary_scopes())

        # Evaluation (EMA) - Exponential moving average
        ema = tf.train.ExponentialMovingAverage(decay=0.998)
        var_list_for_ema = primary_student_variables[0] + primary_student_variables[1]
        ema_op = ema.apply(var_list=var_list_for_ema)
        self.ema_p = self._build_classifier(T.x_tmp, self.num_classes, ema)  # (batch_size, trg_num_classes)

        # teacher_assign_ops = []
        # for t, m in zip(var_teacher, var_main):
        #     ave = ema.average(m)  # compute average ema on each variable of var_main
        #     ave = ave if ave else m
        #     teacher_assign_ops += [tf.assign(t, ave)]  # assign teacher to ave <=> back up h_theta_n-1(x)

        # self.update_teacher = tf.group(*teacher_assign_ops)
        # self.teacher = tb.function(self.tf_session, [T.x_tmp], tf.nn.softmax(teacher_p))

        # EMA accuracy
        self.batch_ema_acc = batch_ema_acc(T.y_tmp, self.ema_p)
        self.fn_batch_ema_acc = tb.function(self.tf_session, [T.x_tmp, T.y_tmp], self.batch_ema_acc)

        # Teacher accuracy
        self.batch_teacher_acc = batch_ema_acc(self.y_trg, self.y_trg_teacher)
        self.fn_batch_teacher_acc = tb.function(self.tf_session, [self.x_trg, self.y_trg, self.is_training], self.batch_teacher_acc)

        # Student accuracy
        self.batch_student_acc = batch_ema_acc(self.y_trg, self.y_trg_logit)
        self.fn_batch_student_acc = tb.function(self.tf_session, [self.x_trg, self.y_trg, self.is_training], self.batch_student_acc)

        # Average accuracy of teacher and student
        self.batch_teac_stud_avg_acc = batch_teac_stud_avg_acc(self.y_trg, self.y_trg_logit, self.y_trg_teacher)
        self.fn_batch_avg_acc = tb.function(self.tf_session, [self.x_trg, self.y_trg, self.is_training], self.batch_teac_stud_avg_acc)

        # Accuracy relies on entropy of each sample in teacher and student
        self.batch_teac_stud_ent_acc = batch_teac_stud_ent_acc(self.y_trg, self.y_trg_logit, self.y_trg_teacher)
        self.fn_batch_ent_acc = tb.function(self.tf_session, [self.x_trg, self.y_trg, self.is_training], self.batch_teac_stud_ent_acc)

        teacher_variables = self._get_variables(self._get_teacher_scopes())
        self.train_teacher = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.phase1_loss,
                                                                     var_list=teacher_variables)
        # original
        self.train_student_main = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.phase2_loss,
                                                                     var_list=teacher_variables + [primary_student_variables[1]])
        self.primary_train_student_op = tf.group(self.train_student_main, ema_op)

        # VA
        self.train_student_main_global = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.phase2_loss_global,
                                                                     var_list=teacher_variables + [primary_student_variables[1]])
        self.primary_train_student_op_global = tf.group(self.train_student_main_global, ema_op)

        if self.save_grads:
            self.grads_wrt_primary_loss = tf.train.AdamOptimizer(self.learning_rate, 0.5).compute_gradients(
                self.phase1_loss, var_list=primary_student_variables)

        # construct secondary loss
        secondary_variables = self._get_variables(self._get_student_secondary_scopes())
        self.secondary_train_student_op = \
            tf.train.AdamOptimizer(self.learning_rate, 0.5).minimize(self.loss_disc,
                                                                   var_list=secondary_variables)

        if self.save_grads:
            self.grads_wrt_secondary_loss = tf.train.AdamOptimizer(self.learning_rate, 0.5).compute_gradients(
                self.loss_disc, var_list=secondary_variables)

        # assign Gt:= Gs for phase 2
        # trg_variables = primary_student_variables[0]
        # src_variables = teacher_variables[0]
        # print('trg_variables', len(trg_variables), 'src_variables', len(src_variables))
        # # trg_variables = primary_student_variables[0] + primary_student_variables[1]
        # # src_variables = teacher_variables[0] + teacher_variables[1][:16]
        # trg_assign_ops = []
        # for trg, src in zip(trg_variables, src_variables):
        #     trg_assign_ops += [tf.assign(trg, src)]
        #
        # self.update_trg_var = tf.group(*trg_assign_ops)

        # summaries
        # tf.summary.scalar('loss/phase1_loss', self.phase1_loss)
        tf.summary.scalar('loss/phase2_loss', self.phase2_loss)
        tf.summary.scalar('loss/phase2_loss_global', self.phase2_loss_global)
        tf.summary.scalar('loss/mimic_loss_ht_g_xs', self.mimic_loss_ht_g_xs)
        tf.summary.scalar('loss/mimic_loss_ht_g_xt', self.mimic_loss_ht_g_xt)
        tf.summary.scalar('loss/generator', self.loss_gen)
        tf.summary.scalar('loss/discriminator', self.loss_disc)
        tf.summary.scalar('loss/src_loss_discriminator', self.src_loss_discriminator)
        tf.summary.scalar('loss/trg_loss_cond_entropy', self.trg_loss_cond_entropy)

        for i in range(self.data_loader.num_src_domain):
            tf.summary.scalar('loss/src_loss_class_{}'.format(i), self.src_loss_class_lst[i])
            tf.summary.scalar('acc/src_acc_{}'.format(i), self.src_accuracy_lst[i])
        tf.summary.scalar('acc/src_domain_acc', self.src_domain_acc)
        tf.summary.scalar('acc/trg_acc', self.trg_accuracy)
        tf.summary.scalar('acc/src_teacher_accuracy', self.src_teacher_accuracy)
        tf.summary.scalar('acc/trg_teacher_accuracy', self.trg_teacher_accuracy)

        tf.summary.scalar('trg_loss_class', self.trg_loss_class)
        tf.summary.scalar('hyperparameters/learning_rate', self.learning_rate)
        tf.summary.scalar('hyperparameters/src_class_trade_off', self.src_class_trade_off)
        tf.summary.scalar('hyperparameters/domain_trade_off', self.domain_trade_off)
        tf.summary.scalar('hyperparameters/gen_trade_off', self.gen_trade_off)
        # tf.summary.scalar('hyperparameters/src_vat_trade_off', self.src_vat_trade_off)
        tf.summary.scalar('hyperparameters/trg_trade_off', self.trg_trade_off)
        tf.summary.scalar('hyperparameters/trg_ent_src_domain_trdoff', self.trg_ent_src_domain_trdoff)

        self.list_str_variables = OrderedDict()
        self.gradient_and_value = []
        if self.save_grads:
            with tf.name_scope("visualize"):
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name + '/values', var)
                for grad, var in self.grads_wrt_primary_loss:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/grads_wrt_primary_loss', grad)
                        self.gradient_and_value += [(grad, var)]
                        self.list_str_variables[var.op.name] = grad.get_shape().as_list()
                for grad, var in self.grads_wrt_secondary_loss:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/grads_wrt_secondary_loss', grad)
        self.tf_merged_summaries = tf.summary.merge_all()

    def mini_batch_class_balanced(self, label, sample_size=20, shuffle=False):
        ''' sample the mini-batch with class balanced
        '''
        label = np.argmax(label, axis=1)
        if shuffle:
            rindex = np.random.permutation(len(label))
            label = label[rindex]

        n_class = len(np.unique(label))
        index = []
        for i in range(n_class):
            s_index = np.nonzero(label == i)

            if len(s_index[0]) < sample_size:
                s_ind = np.random.choice(s_index[0], sample_size)
            else:
                s_ind = np.random.permutation(s_index[0])

            index = np.append(index, s_ind[0:sample_size])
            #          print(index)
        index = np.array(index, dtype=int)
        return index

    def _fit_loop(self):
        print("")
        print('============ LOG-ID: %s ============' % self.current_time)
        mmd_lst = []
        iter_lst = []
        num_src_samples_lst = []
        for k in range(self.data_loader.num_src_domain):
            num_src_samples = self.data_loader.src_train[k][2].shape[0]
            num_src_samples_lst.append(num_src_samples)

        num_trg_samples = self.data_loader.trg_train[0][1].shape[0]
        src_batchsize = self.batch_size // self.data_loader.num_src_domain

        self.tf_session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=101)
        self.log_path = os.path.join(model_dir(), self.model_name, "logs",
                                     "{}_{}_{}".format(self.current_time, self.data_loader.trg_domain_name[0], self.process))
        self.tf_summary_writer = tf.summary.FileWriter(self.log_path, self.tf_session.graph)

        self.checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model", "{}".format(self.mdaot_model_id))
        # print(self.checkpoint_path)
        check_point = tf.train.get_checkpoint_state(self.checkpoint_path)

        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("Load model parameters from %s\n" % check_point.model_checkpoint_path)
            saver.restore(self.tf_session, check_point.model_checkpoint_path)

        feed_y_src_domain = to_categorical(np.repeat(np.arange(self.data_loader.num_src_domain),
                                      repeats=self.sample_size * self.num_classes, axis=0))

        compute_mmd = False
        if compute_mmd:
            self.init_mmd(self.x_trg_mid.shape)
        for it in range(self.num_iters):
            idx_trg_samples = np.random.permutation(num_trg_samples)[:self.batch_size]
            feed_data = dict()
            for k in range(self.data_loader.num_src_domain):

                idx_src_samples = np.random.permutation(num_src_samples_lst[k])[:src_batchsize]
                feed_data[self.x_src_lst[k]] = self.data_loader.src_train[k][1][idx_src_samples, :]
                feed_data[self.y_src_lst[k]] = self.data_loader.src_train[k][2][idx_src_samples]
            feed_data[self.x_trg] = self.data_loader.trg_train[0][1][idx_trg_samples, :]

            feed_data[self.y_src_domain] = feed_y_src_domain
            feed_data[self.is_training] = True

            if it < self.phase1_iters: # train phase 1
                # feed_data[self.phase1_ph] = True
                _, primary_loss, src_loss_class_sum, src_loss_class_lst, src_loss_discriminator, trg_loss_cond_entropy, loss_gen, loss_disc, trg_loss_class, src_acc_lst, trg_acc, src_domain_acc, src_teacher_accuracy, trg_teacher_accuracy, mimic_loss = \
                    self.tf_session.run(
                        [self.train_teacher, self.phase2_loss, self.src_loss_class_sum, self.src_loss_class_lst,
                         self.src_loss_discriminator, self.trg_loss_cond_entropy, self.loss_gen, self.loss_disc,
                         self.trg_loss_class, self.src_accuracy_lst, self.trg_accuracy, self.src_domain_acc, self.src_teacher_accuracy,
                         self.trg_teacher_accuracy, self.mimic_loss_ht_g_xs
                         ],
                        feed_dict=feed_data
                    )
            else:
                if self.process == "origin":
                    _, loss_disc = \
                        self.tf_session.run(
                            [self.secondary_train_student_op, self.loss_disc],
                            feed_dict=feed_data
                        )
                    # feed_data[self.phase1_ph] = False
                    _, primary_loss, src_loss_class_sum, src_loss_class_lst, src_loss_discriminator, trg_loss_cond_entropy, loss_gen, trg_loss_class, src_acc_lst, trg_acc, src_domain_acc, src_teacher_accuracy, trg_teacher_accuracy, mimic_loss_ht_g_xs = \
                        self.tf_session.run(
                            [self.primary_train_student_op, self.phase2_loss, self.src_loss_class_sum, self.src_loss_class_lst, self.src_loss_discriminator, self.trg_loss_cond_entropy, self.loss_gen,
                             self.trg_loss_class, self.src_accuracy_lst, self.trg_accuracy, self.src_domain_acc, self.src_teacher_accuracy, self.trg_teacher_accuracy, self.mimic_loss_ht_g_xs
                        ],
                            feed_dict=feed_data
                        )
                else:
                    _, loss_disc = \
                        self.tf_session.run(
                            [self.secondary_train_student_op, self.loss_disc],
                            feed_dict=feed_data
                        )
                    _, primary_loss, src_loss_class_sum, \
                    src_loss_class_lst, src_loss_discriminator, trg_loss_cond_entropy, \
                    loss_gen, src_acc_lst, \
                    mimic_loss_ht_g_xs, mimic_loss_ht_g_xt = \
                        self.tf_session.run(
                            [self.primary_train_student_op_global, self.phase2_loss, self.src_loss_class_sum,
                             self.src_loss_class_lst, self.src_loss_discriminator, self.trg_loss_cond_entropy,
                             self.loss_gen,
                             self.src_accuracy_lst,
                             self.mimic_loss_ht_g_xs, self.mimic_loss_ht_g_xt
                             ],
                            feed_dict=feed_data
                        )

            if it == 0 or (it + 1) % self.summary_freq == 0:
                if self.process == 'origin':
                    print(
                        "iter %d/%d \n\tprimary_loss %.3f\n\tsrc_loss_class_sum %.3f\n\tloss_disc %.3f\n\tloss_gen %.3f\n\tsrc_loss_discriminator %.3f\n\ttrg_loss_cond_entropy %.3f\n\tmimic_loss_ht_g_xs %.3f \n\ttrg_loss_class %.3f" % (
                            it + 1, self.num_iters, primary_loss, src_loss_class_sum, loss_disc, loss_gen,
                            src_loss_discriminator, trg_loss_cond_entropy, mimic_loss_ht_g_xs, trg_loss_class))
                else:
                    print(
                        "iter %d/%d \n\tprimary_loss %.3f\n\tsrc_loss_class_sum %.3f\n\tloss_disc %.3f\n\tloss_gen %.3f\n\tsrc_loss_discriminator %.3f\n\ttrg_loss_cond_entropy %.3f\n\tmimic_loss_ht_g_xs %.3f \tmimic_loss_ht_g_xt %.3f" % (
                            it + 1, self.num_iters, primary_loss, src_loss_class_sum, loss_disc, loss_gen,
                            src_loss_discriminator, trg_loss_cond_entropy, mimic_loss_ht_g_xs, mimic_loss_ht_g_xt))

                for k in range(self.data_loader.num_src_domain):
                    print('\tsrc_loss_class_{}: {:.3f}\tacc: {:.2f}'.format(self.data_loader.src_train[k][0], src_loss_class_lst[k], src_acc_lst[k]*100))

                y_trg_src_domains_prob, trg_loss_entropy_src_domains = self.tf_session.run([self.y_trg_src_domains_prob, self.trg_loss_entropy_src_domains],
                                                                feed_dict=feed_data)
                y_src_discriminator_prediction = np.argmax(y_trg_src_domains_prob, axis=1)  # (bs,)
                count = []
                count.extend(Counter(y_src_discriminator_prediction).most_common())
                print('\t***Check trg on source domain discriminator')
                print(count)
                print('\tSample pred ', y_trg_src_domains_prob[0])
                print("\tmean(trg_entropy_src_domains) {:.5f}".format(trg_loss_entropy_src_domains))


            if it == 0 or (it + 1) % self.summary_freq == 0:
                print("log_model:", self.log_path)
                if it < 1000:
                    iter_lst.append(it/1000)
                else:
                    iter_lst.append(it//1000)
                # iter_lst.append(it)
                latent_src_lst = self.test_src(self.data_loader.src_test, save_latent=True)
                latent_trg = self.test_trg(self.data_loader.trg_test[0][1], self.data_loader.trg_test[0][2], save_latent=True)

                if not self.only_save_final_model:
                    self.save_trained_model(saver, it + 1)
                elif it + 1 == self.phase1_iters or it + 1 == self.num_iters:
                    self.save_trained_model(saver, it + 1)
                if it >= self.phase1_iters: # and (it + 1) % (self.num_iters // 50) == 0:
                    # Save acc values
                    self.save_value(step=it + 1)

    def test_src(self, data_test_src, save_latent=False, **kwargs):
        print("TEST SRC")
        latent_src_lst = []
        for domain_idx in range(self.data_loader.num_src_domain):
            src_name = data_test_src[domain_idx][0]
            x_test = data_test_src[domain_idx][1]
            y_test = data_test_src[domain_idx][2]

            num_tests = x_test.shape[0]
            y_test_pred = np.ones(num_tests, dtype=int)
            latent_pred = []
            # domain_pred = np.ones(num_tests, dtype=int)

            num_padding_test = int(np.ceil(num_tests / self.batch_size) * self.batch_size)
            idx_test = np.arange(0, num_tests)
            acc_label = []

            for it in range(0, num_padding_test, self.batch_size):
                idx_batch = idx_test[it:it + self.batch_size]
                x_test_batch = x_test[idx_batch, :]
                y_test_batch = y_test[idx_batch]

                feed_data = dict()
                feed_data[self.x_src_lst[domain_idx]] = x_test_batch
                feed_data[self.y_src_lst[domain_idx]] = y_test_batch
                feed_data[self.is_training] = False
                y_predict = self.y_src_softmax_lst[domain_idx]
                g_xs = self.x_src_mid_lst[domain_idx]
                acc_label_i = self.src_accuracy_lst[domain_idx]

                y_predict_tmp, latent_src, acc_i = self.tf_session.run([y_predict, g_xs, acc_label_i], feed_dict=feed_data)

                y_test_pred[idx_batch] = y_predict_tmp
                latent_pred.append(latent_src)
                acc_label.append(acc_i*100.0)

            latent_pred = np.concatenate(latent_pred)
            latent_src_lst.append(latent_pred[:90])
            y_test_gt = np.argmax(y_test[:num_tests], -1)
            # acc_src_i = tf.reduce_mean(tf.cast(tf.equal(y_test_gt, y_test_pred), 'float32')) * 100.0
            acc_mean = np.mean(acc_label)
            print("\t", src_name, "acc:", acc_mean)
            data_tsne_src = {}
            for y in range(self.num_classes):
                idx_y = np.argwhere(y_test_gt == y).reshape(-1)
                num_save = min(100, len(idx_y))
                data_tsne_src.update({y: {
                    'latent': latent_pred[idx_y[:num_save]].tolist(),
                    'y_label': y_test_pred[idx_y[:num_save]].tolist(),
                }})
            if save_latent:
                if not os.path.isdir(self.log_path):
                    os.mkdir(self.log_path)

                with open(os.path.join(self.log_path, "src_latent_{}_2.json".format(src_name)), "w") as fo:
                    json.dump(data_tsne_src, fo)
        return latent_src_lst

    def test_trg(self, x_test, y_test, save_latent=False, **kwargs):
        num_tests = x_test.shape[0]

        y_test_pred = np.ones(num_tests, dtype=int)
        latent_pred = []
        loss_trg = []
        domain_pred = []

        num_padding_test = int(np.ceil(num_tests / self.batch_size) * self.batch_size)
        idx_test = np.zeros(num_tests, dtype=int)
        idx_test[0:num_tests] = np.arange(0, num_tests)
        acc_label = []
        size = 0
        for it in range(0, num_padding_test, self.batch_size):
            idx_batch = idx_test[it:it + self.batch_size]
            x_test_batch = x_test[idx_batch, :]
            y_test_batch = y_test[idx_batch]

            feed_data = dict()
            feed_data[self.x_trg] = x_test_batch
            feed_data[self.y_trg] = y_test_batch
            feed_data[self.is_training] = False
            y_predict = self.y_trg_pred
            g_xt = self.x_trg_mid
            c_g_xt = self.y_trg_src_domains_prob
            acc_label_i = self.trg_accuracy

            y_predict_tmp, latent_trg, y_predict_domain, acc_i, loss_i = self.tf_session.run([y_predict, g_xt, c_g_xt, acc_label_i, self.trg_loss_class], feed_dict=feed_data)

            y_test_pred[idx_batch] = y_predict_tmp
            latent_pred.append(latent_trg)
            domain_pred.append(y_predict_domain)
            acc_label.append(acc_i*100.0)
            loss_trg.append(loss_i)

        latent_pred = np.concatenate(latent_pred)
        domain_pred = np.concatenate(domain_pred)

        y_test_gt = np.argmax(y_test[:num_tests], -1)
        # acc_trg = tf.reduce_mean(tf.cast(tf.equal(y_test_gt, y_test_pred), 'float32')) * 100.0
        acc_trg = np.mean(acc_label)
        print("TEST_TRG", "\tloss_trg:", sum(loss_trg)/it, "\tacc trg:", acc_trg)

        data_tsne_trg = {}
        for y in range(self.num_classes):
            idx_y = np.argwhere(y_test_gt == y).reshape(-1)
            num_save = min(100, len(idx_y))
            data_tsne_trg.update({y: {
                'latent': latent_pred[idx_y[:num_save]].tolist(),
                'y_d': domain_pred[idx_y[:num_save]].tolist(),
                'y_label': y_test_pred[idx_y[:num_save]].tolist(),
            }})
        if save_latent:
            if not os.path.isdir(self.log_path):
                os.mkdir(self.log_path)

            with open(os.path.join(self.log_path, "trg_latent_2.json"), "w") as fo:
                json.dump(data_tsne_trg, fo)
        return latent_pred

    def save_trained_model(self, saver, step):
        # Save model
        checkpoint_path = os.path.join(model_dir(), self.model_name, "saved-model",
                                       "{}_{}_{}".format(self.current_time, self.data_loader.trg_domain_name[0], self.process))
        checkpoint_path = os.path.join(checkpoint_path, "stem_" + self.current_time + ".ckpt")

        directory = os.path.dirname(checkpoint_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        saver.save(self.tf_session, checkpoint_path, global_step=step)

    def save_value(self, step):
        # Save ema accuracy
        # acc_teacher_ema, summary_teacher_ema = self.compute_value(self.fn_batch_ema_acc, 'test/teacher_ema_1k',
        #                                                           x_full=self.data_loader.trg_test[0][1],
        #                                                           y=None, labeler=self.teacher, full=False)
        ema_acc, teacher_acc, student_acc, avg_acc, ent_acc, summary = self.compute_value(
                                                                    x_full=self.data_loader.trg_test[0][1],
                                                                    y=self.data_loader.trg_test[0][2], labeler=None)
        # acc_trg_train_ema, summary_trg_train_ema = self.compute_value(self.fn_batch_ema_acc, 'test/trg_train_ema_1k',
        #                                                               x_full=self.x_trg_,
        #                                                               y=self.y_trg_, labeler=None, full=False)

        # self.tf_summary_writer.add_summary(summary_teacher_ema, step)
        self.tf_summary_writer.add_summary(summary, step)
        # self.tf_summary_writer.add_summary(summary_trg_train_ema, step)
        self.tf_summary_writer.flush()

        # print_list = ['teacher_ema_1k', round(acc_teacher_ema*100, 2), 'trg_test_ema', round(acc_trg_test_ema*100, 2), 'trg_train_ema_1k',
        #               round(acc_trg_train_ema*100, 2)]
        print_list = ['ema_acc', round(ema_acc * 100, 2),
                      'teacher_acc', round(teacher_acc * 100, 2),
                      'student_acc', round(student_acc * 100, 2),
                      'avg_acc', round(avg_acc * 100, 2),
                      'ent_acc', round(ent_acc * 100, 2)]
        print(print_list)

    def compute_value(self, x_full, y, labeler, full=True):
        """Compute exactly acc of value w.r.t. data
        fn_batch_ema_acc - (fn) Takes (x, y) as input and returns value
        tag    - (str) summary tag for FileWriter
        data   - (Data) data object with images/labels attributes
        full   - (bool) use full dataset v. first 1000 samples
        """

        # Convert y to one-hot encoder
        # y = self.convert_one_hot(y)

        with tb.nputils.FixedSeed(0):
            shuffle = np.random.permutation(len(x_full))

        xs = x_full[shuffle]
        ys = y[shuffle] if y is not None else None

        if not full:
            xs = xs[:1000]
            ys = ys[:1000] if ys is not None else None

        n = len(xs)
        bs = 200

        ema_acc_full = np.ones(n, dtype=float)
        teacher_acc_full = np.ones(n, dtype=float)
        student_acc_full = np.ones(n, dtype=float)
        avg_acc_full = np.ones(n, dtype=float)
        ent_acc_full = np.ones(n, dtype=float)

        for i in range(0, n, bs):
            x = xs[i:i + bs]
            y = ys[i:i + bs] if ys is not None else labeler(x)
            ema_acc_batch = self.fn_batch_ema_acc(x, y)  # (batch_size,)
            teacher_acc_batch = self.fn_batch_teacher_acc(x, y, False)
            student_acc_batch = self.fn_batch_student_acc(x, y, False)
            avg_acc_batch = self.fn_batch_avg_acc(x, y, False)
            ent_acc_batch = self.fn_batch_ent_acc(x, y, False)

            ema_acc_full[i:i + bs] = ema_acc_batch
            teacher_acc_full[i:i + bs] = teacher_acc_batch
            student_acc_full[i:i + bs] = student_acc_batch
            avg_acc_full[i:i + bs] = avg_acc_batch
            ent_acc_full[i:i + bs] = ent_acc_batch

        ema_acc = np.mean(ema_acc_full)
        teacher_acc = np.mean(teacher_acc_full)
        student_acc = np.mean(student_acc_full)
        avg_acc = np.mean(avg_acc_full)
        ent_acc = np.mean(ent_acc_full)

        summary_1 = tf.Summary.Value(tag='trg_test/ema_acc', simple_value=ema_acc)
        summary_2 = tf.Summary.Value(tag='trg_test/teacher_acc', simple_value=teacher_acc)
        summary_3 = tf.Summary.Value(tag='trg_test/student_acc', simple_value=student_acc)
        summary_4 = tf.Summary.Value(tag='trg_test/avg_acc', simple_value=avg_acc)
        summary_5 = tf.Summary.Value(tag='trg_test/ent_acc', simple_value=ent_acc)

        summary = tf.Summary(value=[summary_1, summary_2, summary_3, summary_4, summary_5])
        return ema_acc, teacher_acc, student_acc, avg_acc, ent_acc, summary

    def init_mmd(self, dim_latent, num_beta=10):
        self.mmd_input1 = tf.placeholder(dtype=tf.float32, shape=[None, dim_latent[1], dim_latent[2], dim_latent[3]],
                                        name='mmd_input1')
        self.mmd_input1_flatten = tf.reshape(self.mmd_input1, [-1, dim_latent[1]*dim_latent[2]*dim_latent[3]])
        self.mmd_input2 = tf.placeholder(dtype=tf.float32, shape=[None, dim_latent[1], dim_latent[2], dim_latent[3]],
                                        name='mmd_input2')
        self.mmd_input2_flatten = tf.reshape(self.mmd_input2, [-1, dim_latent[1]*dim_latent[2]*dim_latent[3]])
        self.mmd_beta_lst = []
        for i in range(num_beta):
            self.mmd_beta_lst.append(
                tf.placeholder(dtype=tf.float32, shape=(), name='beta_{}'.format(i)))

        x1x1_kernel_lst = self.gaussian_kernel_tf(self.mmd_input1_flatten, self.mmd_input1_flatten, self.mmd_beta_lst)
        x1x2_kernel_lst = self.gaussian_kernel_tf(self.mmd_input1_flatten, self.mmd_input2_flatten, self.mmd_beta_lst)
        x2x2_kernel_lst = self.gaussian_kernel_tf(self.mmd_input2_flatten, self.mmd_input2_flatten, self.mmd_beta_lst)

        self.mmd_diff_lst = []
        for x1x1, x1x2, x2x2 in zip(x1x1_kernel_lst, x1x2_kernel_lst, x2x2_kernel_lst):
            self.mmd_diff_lst.append(tf.reduce_mean(x1x1) - 2 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2))

    def gaussian_kernel_tf(self, x1, x2, beta_lst=[]):
        # r = x1.dimshuffle(0, 'x', 1)
        # x1_shape = np.shape(x1)
        r = tf.expand_dims(x1, axis=1)
        # a = K.square(r - x2).sum(axis=-1)
        a = tf.reduce_sum(tf.square(r - x2), axis=-1)
        kernal_lst = []
        for b in beta_lst:
            kernal_lst.append(tf.exp(-b * a))
        return kernal_lst

def mmd(x1, x2, beta_lst=[]):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    if len(beta_lst) == 0:
        beta_lst = [1.0]
    x1x1_lst = gaussian_kernel(x1, x1, beta_lst)
    x1x2_lst = gaussian_kernel(x1, x2, beta_lst)
    x2x2_lst = gaussian_kernel(x2, x2, beta_lst)
    mmd_diff_lst = []
    for x1x1, x1x2, x2x2 in zip(x1x1_lst, x1x2_lst, x2x2_lst):
        mmd_diff_lst.append(np.mean(x1x1) - 2 * np.mean(x1x2) + np.mean(x2x2))
    return mmd_diff_lst


def gaussian_kernel(x1, x2, beta_lst=[]):
    # r = x1.dimshuffle(0, 'x', 1)
    # x1_shape = np.shape(x1)
    r = np.expand_dims(x1, axis=1)
    # a = K.square(r - x2).sum(axis=-1)
    a = np.sum(np.square(r - x2), axis=-1)
    kernal_lst = []
    for b in beta_lst:
        kernal_lst.append(np.exp(-b * a))
    return kernal_lst
