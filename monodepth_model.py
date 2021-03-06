# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.resampler as resampler

from bilinear_sampler import *

monodepth_parameters = namedtuple(
    'parameters',
    'encoder, '
    'height, width, '
    'batch_size, '
    'num_threads, '
    'num_epochs, '
    'do_stereo, '
    'wrap_mode, '
    'use_deconv, '
    'alpha_image_loss, '
    'disp_gradient_loss_weight, '
    'fb_loss_weight, '
    'explainability_loss_weight, '
    'explainability_mask, '
    'color_augmentation_prob, '
    'flip_augmentation_prob, '
    'swap_augmentation_prob, '
    'full_summary')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, first_image, second_image, delta_position, delta_angle, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.first_image = first_image
        self.second_image = second_image
        self.delta_position = delta_position
        self.delta_angle = delta_angle
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_backward(self, img, disp):
        return self.bilinear_sampler(img, -disp)

    def generate_image_forward(self, img, disp):
        return self.bilinear_sampler(img, disp)

    def bilinear_sampler(self, img, disp, name='bilinear_sampler'):
        def _transform(img, disp):
            with tf.variable_scope('transform'):
                # grid of (x_t, y_t, 1), eq (1) in ref [1]
                x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                       tf.linspace(0.0, _height_f - 1.0, _height))
                offsets = tf.concat([x_t[:, :, None], y_t[:, :, None]], axis=-1)

                offset_disparities = disp * [_width_f, _height_f] + offsets[None, :, :, :]

                return resampler.resampler(img, offset_disparities, name='bilinear_sampler')

        with tf.variable_scope(name):
            _num_batch = tf.shape(img)[0]
            _height = tf.shape(img)[1]
            _width = tf.shape(img)[2]
            _num_channels = tf.shape(img)[3]

            _height_f = tf.cast(_height, tf.float32)
            _width_f = tf.cast(_width, tf.float32)

            output = _transform(img, disp)
            return output

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 4, 3, 1, tf.nn.tanh)
        return disp

    def get_explainability(self, x):
        return self.conv(x, 2, 3, 1, tf.nn.sigmoid)

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def posenet(self, conv_shape):
        pose_out = tf.concat([self.delta_position, self.delta_angle], axis=1)
        # pose_out = slim.batch_norm(
        #     angle_pos, scale=True, center=True, is_training=self.mode == 'train')
        pose_out = slim.fully_connected(pose_out, 32, activation_fn=tf.nn.elu)
        pose_out = slim.fully_connected(pose_out, 16, activation_fn=tf.nn.elu)

        # tile into the spatial shape of the convolution:
        pose_tiled = tf.tile(pose_out, [1, conv_shape[1].value * conv_shape[2].value])
        pose_reshaped = tf.reshape(pose_tiled, (conv_shape[0], conv_shape[1], conv_shape[2], 16))

        return pose_reshaped

    def image_weight(self, x, explainability):
        if not self.params.explainability_mask:
            weighted_x = x
        else:
            weighted_x = [x[i] * explainability[i] for i in range(4)]

        return [tf.reduce_mean(val) for val in weighted_x]

    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7) # H/2
            conv2 = self.conv_block(conv1,             64, 5) # H/4
            conv3 = self.conv_block(conv2,            128, 3) # H/8
            conv4 = self.conv_block(conv3,            256, 3) # H/16
            conv5 = self.conv_block(conv4,            512, 3) # H/32
            conv6 = self.conv_block(conv5,            512, 3) # H/64
            conv7 = self.conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('posenet'):
            pose_out = self.posenet(conv7.shape)
            posed_conv7 = tf.concat([conv7, pose_out], axis=3)
            # use a 1x1 convolution to resize back to normal
            posed_conv7 = conv(posed_conv7, 512, 1, 1)
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(posed_conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

        if self.params.explainability_mask:
            with tf.variable_scope("explainability"):
                self.explain1 = self.get_explainability(iconv1)
                self.explain2 = self.get_explainability(iconv2)
                self.explain3 = self.get_explainability(iconv3)
                self.explain4 = self.get_explainability(iconv4)


    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        with tf.variable_scope('posenet'):
            pose_out = self.posenet(conv5.shape)
            posed_conv5 = tf.concat([conv5, pose_out], axis=3)
            # use a 1x1 convolution to resize back to normal
            posed_conv5 = conv(posed_conv5, 512, 1, 1)
        
        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(posed_conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

        if self.params.explainability_mask:
            with tf.variable_scope("explainability"):
                self.explain1 = self.get_explainability(iconv1)
                self.explain2 = self.get_explainability(iconv2)
                self.explain3 = self.get_explainability(iconv3)
                self.explain4 = self.get_explainability(iconv4)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.first_pyramid  = self.scale_pyramid(self.first_image, 4)
                if self.mode == 'train':
                    self.second_pyramid = self.scale_pyramid(self.second_image, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.first_image, self.second_image], 3)
                else:
                    self.model_input = self.first_image

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_backward_est  = [d[:, :, :, 0:2] for d in self.disp_est]
            self.disp_forward_est = [d[:, :, :, 2:4] for d in self.disp_est]

        # EXPLAINABILITY MASK
        with tf.variable_scope('explainability'):
            if self.params.explainability_mask:
                self.explain_est = [self.explain1, self.explain2, self.explain3, self.explain4]
            else:
                self.explain_est = []

            self.explain_backward_est = [e[:, :, :, 0:1] for e in self.explain_est]
            self.explain_forward_est  = [e[:, :, :, 1:2] for e in self.explain_est]

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            if hasattr(self, 'first_pyramid'):
                self.second_est = [
                    self.generate_image_forward(self.first_pyramid[i], self.disp_forward_est[i])
                    for i in range(4)
                ]
            if hasattr(self, 'second_pyramid'):
                self.first_est = [
                    self.generate_image_backward(self.second_pyramid[i], self.disp_backward_est[i])
                    for i in range(4)
                ]

        if self.mode == 'test':
            return

        # FB CONSISTENCY
        with tf.variable_scope('forward-backward'):
            self.second_to_first_disp = [
                self.generate_image_backward(self.disp_forward_est[i], self.disp_backward_est[i])
                for i in range(4)
            ]
            self.first_to_second_disp = [
                self.generate_image_forward(self.disp_backward_est[i], self.disp_forward_est[i])
                for i in range(4)
            ]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_first_smoothness  = self.get_disparity_smoothness(self.disp_backward_est, self.first_pyramid)
            self.disp_second_smoothness = self.get_disparity_smoothness(self.disp_forward_est, self.second_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_first = [tf.abs(self.first_est[i] - self.first_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_first  = self.image_weight(self.l1_first, self.explain_backward_est)
            self.l1_second = [tf.abs(self.second_est[i] - self.second_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_second = self.image_weight(self.l1_second, self.explain_forward_est)

            # SSIM
            self.ssim_first = [self.SSIM(self.first_est[i], self.first_pyramid[i]) for i in range(4)]
            self.ssim_loss_first  = self.image_weight(self.ssim_first, self.explain_backward_est)
            self.ssim_second = [self.SSIM(self.second_est[i], self.second_pyramid[i]) for i in range(4)]
            self.ssim_loss_second = self.image_weight(self.ssim_second, self.explain_forward_est)

            # WEIGHTED SUM
            self.image_loss_second = [
                self.params.alpha_image_loss * self.ssim_loss_second[i] +
                (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_second[i]
                for i in range(4)
            ]
            self.image_loss_first  = [
                self.params.alpha_image_loss * self.ssim_loss_first[i] +
                (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_first[i]
                for i in range(4)
            ]
            self.image_loss = tf.add_n(self.image_loss_first + self.image_loss_second)

            # DISPARITY SMOOTHNESS
            self.disp_first_loss  = [tf.reduce_mean(tf.abs(self.disp_first_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_second_loss = [tf.reduce_mean(tf.abs(self.disp_second_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_first_loss + self.disp_second_loss)

            # FORWARD-BACKWARD CONSISTENCY
            self.fb_first_loss  = [tf.reduce_mean(tf.abs(self.second_to_first_disp[i] - self.disp_backward_est[i])) for i in range(4)]
            self.fb_second_loss = [tf.reduce_mean(tf.abs(self.first_to_second_disp[i] - self.disp_forward_est[i])) for i in range(4)]
            self.fb_loss = tf.add_n(self.fb_first_loss + self.fb_second_loss)

            # EXPLAINABILITY LOSS
            if self.params.explainability_mask:
                # cross-entropy where the label is 1 (totally explainable) for every pixel
                self.explain_forward_loss = [
                    -tf.reduce_mean(tf.log(self.explain_forward_est[i])) / 2 ** i
                    for i in range(4)
                ]
                self.explain_backward_loss = [
                    -tf.reduce_mean(tf.log(self.explain_backward_est[i])) / 2 ** i
                    for i in range(4)
                ]
                self.explainability_loss = tf.add_n(self.explain_forward_loss + self.explain_backward_loss)
            else:
                self.explainability_loss = 0.

            # TOTAL LOSS
            self.total_loss = \
                self.image_loss + \
                self.params.disp_gradient_loss_weight * self.disp_gradient_loss + \
                self.params.fb_loss_weight * self.fb_loss + \
                self.params.explainability_loss_weight * self.explainability_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_first[i] + self.ssim_loss_second[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_first[i] + self.l1_reconstruction_loss_second[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_first[i] + self.image_loss_second[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_first_loss[i] + self.disp_second_loss[i], collections=self.model_collection)
                tf.summary.scalar('fb_loss_' + str(i), self.fb_first_loss[i] + self.fb_second_loss[i], collections=self.model_collection)
                if self.params.explainability_mask:
                    tf.summary.scalar('explain_loss_' + str(i), self.explain_forward_loss[i] + self.explain_backward_loss[i], collections=self.model_collection)

                disparity_first_abs = tf.reduce_sum(
                    self.disp_backward_est[i] ** 2, axis=3, keepdims=True) ** 0.5
                disparity_second_abs = tf.reduce_sum(
                    self.disp_forward_est[i] ** 2, axis=3, keepdims=True) ** 0.5
                tf.summary.image('disp_first_est_' + str(i), disparity_first_abs, max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_second_est_' + str(i), disparity_second_abs, max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('first_est_' + str(i), self.first_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('second_est_' + str(i), self.second_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_first_' + str(i), self.ssim_first[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_second_' + str(i), self.ssim_second[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_first_' + str(i), self.l1_first[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_second_' + str(i), self.l1_second[i], max_outputs=4, collections=self.model_collection)

                    if self.params.explainability_mask:
                        tf.summary.image('explain_first_' + str(i), self.explain_backward_est[i], max_outputs=4, collections=self.model_collection)
                        tf.summary.image('explain_second_' + str(i), self.explain_forward_est[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('first', self.first_image, max_outputs=4, collections=self.model_collection)
                tf.summary.image('second', self.second_image, max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.histogram("delta_position_f", self.delta_position[:, 0], collections=self.model_collection)
                tf.summary.histogram("delta_position_l", self.delta_position[:, 1], collections=self.model_collection)
                tf.summary.histogram("delta_position_u", self.delta_position[:, 2], collections=self.model_collection)

                tf.summary.histogram("delta_angle_f", self.delta_angle[:, 0], collections=self.model_collection)
                tf.summary.histogram("delta_angle_l", self.delta_angle[:, 1], collections=self.model_collection)
                tf.summary.histogram("delta_angle_u", self.delta_angle[:, 2], collections=self.model_collection)

