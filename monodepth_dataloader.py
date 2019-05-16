# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test' and not self.params.do_stereo:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            left_image_o  = self.read_image(left_image_path)
        else:
            left_image_path  = tf.string_join([self.data_path, split_line[0]])
            right_image_path = tf.string_join([self.data_path, split_line[1]])
            left_image_o  = self.read_image(left_image_path)
            right_image_o = self.read_image(right_image_path)

        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            left_image  = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image), lambda: (left_image, right_image))

            left_image.set_shape( [None, None, 3])
            right_image.set_shape([None, None, 3])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            self.left_image_batch, self.right_image_batch = tf.train.shuffle_batch([left_image, right_image],
                        params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            self.left_image_batch.set_shape( [2, None, None, 3])

            if self.params.do_stereo:
                self.right_image_batch = tf.stack([right_image_o,  tf.image.flip_left_right(right_image_o)],  0)
                self.right_image_batch.set_shape( [2, None, None, 3])

    def augment_image_pair(self, left_image, right_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = tf.strings.length(image_path)
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) // 5
            image  =  image[:crop_height,:,:]

        image  = tf.image.convert_image_dtype(image,  tf.float32)
        image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image


class TemporalDepthDataloader(MonodepthDataloader):
    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode

        self.left_image_batch  = None
        self.right_image_batch = None
        self.delta_position = None
        self.delta_angle = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)
        split_line = tf.string_split([line]).values

        first_image_path = tf.string_join([self.data_path, split_line[0]])
        second_image_path = tf.string_join([self.data_path, split_line[2]])
        first_oxts_path = tf.string_join([self.data_path, split_line[1]])
        second_oxts_path = tf.string_join([self.data_path, split_line[3]])
        timestamp = tf.strings.to_number(split_line[4], out_type=tf.dtypes.float32)

        # we load only one image for test, except if we trained a stereo model
        if mode == 'test':
            first_image = self.read_image(first_image_path)
        else:
            first_image = self.read_image(first_image_path)
            second_image = self.read_image(second_image_path)

            first_oxts = self.read_oxts(first_oxts_path)
            second_oxts = self.read_oxts(second_oxts_path)

            delta_position, delta_angle = self.compute_delta_position_angle(first_oxts, second_oxts, timestamp)


        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            first_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(first_image), lambda: first_image)
            second_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(second_image),  lambda: second_image)
            delta_position = tf.cond(do_flip > 0.5, lambda: delta_position * [1., -1., 1.], lambda: delta_position)
            delta_angle = tf.cond(do_flip > 0.5, lambda: delta_angle * [1., -1., 1.], lambda: delta_angle)

            # randomly swap images
            do_swap = tf.random_uniform([], 0, 1)
            first_image = tf.cond(do_swap > 0.5, lambda: second_image, lambda: first_image)
            second_image = tf.cond(do_swap > 0.5, lambda: first_image,  lambda: second_image)
            delta_position = tf.cond(do_swap > 0.5, lambda: delta_position * -1., lambda: delta_position)
            delta_angle = tf.cond(do_swap > 0.5, lambda: delta_angle * -1., lambda: delta_angle)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            first_image, second_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(first_image, second_image), lambda: (first_image, second_image))

            first_image.set_shape( [None, None, 3])
            second_image.set_shape([None, None, 3])
            delta_position.set_shape([3])
            delta_angle.set_shape([3])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + 4 * params.batch_size
            # self.first_image_batch, self.first_oxts_batch, self.second_image_batch, self.second_oxts_batch = \
            #     first_image[None, :], first_oxts[None, :], second_image[None, :], second_oxts[None, :]
            self.first_image_batch, self.second_image_batch, self.delta_position, self.delta_angle = \
                tf.train.shuffle_batch(
                    [first_image, second_image, delta_position, delta_angle],
                    params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            self.left_image_batch = tf.stack([first_image,  tf.image.flip_left_right(first_image)],  0)
            self.left_image_batch.set_shape( [2, None, None, 3])
            self.delta_position = tf.constant([[0., 0., 0.]] * 2)
            self.delta_angle = tf.constant([[0., 0., 0.]] * 2)

            if self.params.do_stereo:
                self.right_image_batch = tf.stack([second_image,  tf.image.flip_left_right(second_image)],  0)
                self.right_image_batch.set_shape( [2, None, None, 3])

    def read_oxts(self, oxts_path):
        fs = tf.io.read_file(oxts_path)
        oxts = tf.io.decode_csv(
            fs,
            record_defaults=[[float(0)]] * 9,
            field_delim=" ",
            select_cols=[8, 9, 10, 14, 15, 16, 20, 21, 22])
        return tf.stack(oxts)

    def compute_delta_position_angle(self, first_oxts, second_oxts, delta_time):
        oxts = (first_oxts + second_oxts) / 2.

        delta_position = oxts[0:3] * delta_time + oxts[3:6] * (delta_time ** 2)
        delta_angle = oxts[6:9] * delta_time

        return delta_position, delta_angle
