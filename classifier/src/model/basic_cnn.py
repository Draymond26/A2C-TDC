# -*- encoding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from baselines.classifier.src.layer.conv_layer import ConvLayer
from baselines.classifier.src.layer.dense_layer import DenseLayer
from baselines.classifier.src.layer.pool_layer import PoolLayer

class ConvNet():

    def __init__(self,
                 n_channel,
                 n_classes,
                 image_height,
                 image_width):

        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[None, image_height, image_width, n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.int64, shape=[None], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')

        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(None, image_height, image_width, n_channel), n_size=3, n_filter=64,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')

        conv_layer2 = ConvLayer(
            input_shape=(None, int(image_height/2), int(image_width/2), 64), n_size=3, n_filter=128,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        '''
        conv_layer3 = ConvLayer(
            input_shape=(None, int(image_height/4), int(image_width/4), 128), n_size=3, n_filter=256,
            stride=1, activation='relu', batch_normal=False, weight_decay=1e-4,
            name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        '''
        dense_layer1 = DenseLayer(
            input_shape=(None, int(image_height/4) * int(image_width/4) * 128), hidden_dim=1024,
            activation='relu', dropout=True, keep_prob=self.keep_prob,
            batch_normal=False, weight_decay=1e-4, name='dense1')

        dense_layer2 = DenseLayer(
            input_shape=(None, 1024), hidden_dim=n_classes,
            activation='none', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=1e-4, name='dense2')

        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        '''
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv3)
        '''
        input_dense1 = tf.reshape(hidden_pool2, [-1, int(image_height/4) * int(image_width/4) * 128])
        output_dense1 = dense_layer1.get_output(input=input_dense1)
        logits = dense_layer2.get_output(input=output_dense1)
        self.logit = tf.argmax(logits, 1, name='predicted_class')

        # 目标函数
        self.objective = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels))
        tf.add_to_collection('losses', self.objective)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        # 优化器
        lr = tf.cond(tf.less(self.global_step, 20),
                     lambda: tf.constant(0.01),
                     lambda: tf.cond(tf.less(self.global_step, 100),
                                     lambda: tf.constant(0.001),
                                     lambda: tf.constant(0.0001)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr*0.001).minimize(
            self.avg_loss, global_step=self.global_step)

        # 观察值
        correct_prediction = tf.equal(self.labels, self.logit)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))