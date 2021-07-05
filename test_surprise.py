import tensorflow as tf
import numpy as np
import os
import time
import argparse
import librosa
import sys
from utils import *
from ops import *
import librosa.display
from IPython.display import Audio
# import matplotlib
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
import soundfile as sf

#%matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (16, 4)

random_seed = 0
np.random.seed(random_seed)


def Style_Encoder(inputs, style_dim=16, reuse=False,
                  scope='style_encoder'):  # [1, 24, 128] = [batch_size, feature_channel, time]

    inputs = tf.transpose(inputs, perm=[0, 2, 1],
                          name='input_transpose')  # [1, 128, 24] = [batch_size, time, feature_channel]

    with tf.variable_scope(scope, reuse=reuse):
        h1 = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv')  # [1, 128, 128]
        h1_gates = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample1d_block_withoutIN(inputs=h1_glu, filters=256, kernel_size=5, strides=2,
                                          name_prefix='downsample1d_block1')  # [1, 64, 256]
        d2 = downsample1d_block_withoutIN(inputs=d1, filters=512, kernel_size=5, strides=2,
                                          name_prefix='downsample1d_block2')  # [1, 32, 512]

        d3 = downsample1d_block_withoutIN(inputs=d2, filters=512, kernel_size=3, strides=2,
                                          name_prefix='downsample1d_block3')  # [1, 16, 512]
        d4 = downsample1d_block_withoutIN(inputs=d3, filters=512, kernel_size=3, strides=2,
                                          name_prefix='downsample1d_block4')  # [1, 8, 512]

        # Global Average Pooling
        p1 = adaptive_avg_pooling(d4)  # [1, 1, 512]
        style = conv1d_layer(inputs=p1, filters=style_dim, kernel_size=1, strides=1, name='SE_logit')  # [1, 1, 16]

        return style  # [1, 1, 16]


def Content_Encoder(inputs, reuse=False, scope='content_encoder'):
    # IN removes the original feature mean and variance that represent important style information
    inputs = tf.transpose(inputs, perm=[0, 2, 1],
                          name='input_transpose')  # [1, 24, 128] = [batch_size, time, feature_channel]

    with tf.variable_scope(scope, reuse=reuse):
        h1 = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv')  # [1, 128, 128]
        h1_norm = instance_norm_layer(inputs=h1, name='h1_norm')
        h1_gates = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_gates')
        h1_norm_gates = instance_norm_layer(inputs=h1_gates, name='h1_norm_gates')
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name='h1_glu')

        # downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=256, kernel_size=5, strides=2,
                                name_prefix='downsample1d_block1')  # [1, 64, 256]
        d2 = downsample1d_block(inputs=d1, filters=512, kernel_size=5, strides=2,
                                name_prefix='downsample1d_block2')  # [1, 32, 512]

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, kernel_size=3, strides=1,
                              name_prefix='residual1d_block1')  # [1, 32, 512]
        r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3, strides=1, name_prefix='residual1d_block2')
        r3 = residual1d_block(inputs=r2, filters=512, kernel_size=3, strides=1, name_prefix='residual1d_block3')
        content = residual1d_block(inputs=r3, filters=512, kernel_size=3, strides=1, name_prefix='residual1d_block4')

        return content  # [1, 32, 512]


def MLP(style, reuse=False, scope='MLP'):  # [1, 1, 16]

    with tf.variable_scope(scope, reuse=reuse):
        x1 = linear(style, 512, scope='linear_1')  # [1, 1, 512]
        x1_gates = linear(x1, 512, scope='linear_1_gates')
        x1_glu = gated_linear_layer(inputs=x1, gates=x1_gates, name='x1_glu')

        x2 = linear(x1_glu, 512, scope='linear_2')
        x2_gates = linear(x2, 512, scope='linear_2_gates')
        x2_glu = gated_linear_layer(inputs=x2, gates=x2_gates, name='x2_glu')

        mu = linear(x2_glu, 512, scope='mu')
        sigma = linear(x2_glu, 512, scope='sigma')

        mu = tf.reshape(mu, shape=[-1, 1, 512])  # [1, 1, 512]
        sigma = tf.reshape(sigma, shape=[-1, 1, 512])  # [1, 1, 512]

        return mu, sigma  # [1, 1, 512]


def Decoder(content, style, reuse=False, scope="decoder"):
    with tf.variable_scope(scope, reuse=reuse):
        mu, sigma = MLP(style, reuse)  # [1, 1, 512]
        x = content  # [1, 32, 512]

        # Adaptive Residual blocks
        r1 = residual1d_block_adaptive(inputs=x, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1,
                                       name_prefix='residual1d_block1')  # [1, 32, 512]
        r2 = residual1d_block_adaptive(inputs=r1, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1,
                                       name_prefix='residual1d_block2')
        r3 = residual1d_block_adaptive(inputs=r2, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1,
                                       name_prefix='residual1d_block3')

        # Upsample
        u1 = upsample1d_block(inputs=r3, filters=512, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample1d_block1')  # [1, 64, 512]
        u2 = upsample1d_block(inputs=u1, filters=256, kernel_size=5, strides=1, shuffle_size=2,
                              name_prefix='upsample1d_block2')  # [1, 128, 256]

        # Output
        o1 = conv1d_layer(inputs=u2, filters=24, kernel_size=15, strides=1, name='o1_conv')  # [1, 128, 24]
        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')  # [1, 24, 128]

        return o2  # [1, 24, 128] = [batch_size, feature_channel, time]


def Discriminator(inputs, reuse=False, scope='discriminator'):

    # inputs = [batch_size, num_features, time]
    # add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)                                                                                                 # [1, 24, 128, 1]

    with tf.variable_scope(scope, reuse=reuse):

        h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], name='h1_conv')                               # [1, 24, 64, 128]
        h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block1')      # [1, 12, 32, 256]
        d2 = downsample2d_block(inputs=d1, filters=512, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block2')          # [1, 6, 16, 512]
        d3 = downsample2d_block(inputs=d2, filters=1024, kernel_size=[6, 3], strides=[1, 2], name_prefix='downsample2d_block3')         # [1, 6, 8, 1024]

        # Output
        o1 = tf.layers.dense(inputs=d3, units=1, activation=tf.nn.sigmoid)

        return [o1]                                                                                                                       # [1, 6, 8, 1]


##################################################################################
# Model
##################################################################################

def Encoder_A(x_A, reuse=False):
    style_A = Style_Encoder(x_A, reuse=reuse, scope='style_encoder_A')
    content_A = Content_Encoder(x_A, reuse=reuse, scope='content_encoder_A')

    return content_A, style_A

def Encoder_B(x_B, reuse=False):
    style_B = Style_Encoder(x_B, reuse=reuse, scope='style_encoder_B')
    content_B = Content_Encoder(x_B, reuse=reuse, scope='content_encoder_B')

    return content_B, style_B

def Decoder_A(content_B, style_A, reuse=False):
    x_ba = Decoder(content=content_B, style=style_A, reuse=reuse, scope='decoder_A')

    return x_ba

def Decoder_B(content_A, style_B, reuse=False):
    x_ab = Decoder(content=content_A, style=style_B, reuse=reuse, scope='decoder_B')

    return x_ab

def discriminate_real(x_A, x_B):
    real_A_logit = Discriminator(x_A, scope="discriminator_A")
    real_B_logit = Discriminator(x_B, scope="discriminator_B")

    return real_A_logit, real_B_logit

def discriminate_fake(x_ba, x_ab):
    fake_A_logit = Discriminator(x_ba, reuse=True, scope="discriminator_A")
    fake_B_logit = Discriminator(x_ab, reuse=True, scope="discriminator_B")

    return fake_A_logit, fake_B_logit


class EmoMUNIT(object):
    def __init__(self, sess):

        self.train_A_dir = 'neutral'
        self.train_B_dir = 'surprise'
        self.validation_A_dir = 'test_neutral' #output gtts
        # self.validation_B_dir = '/content/drive/MyDrive/EmoMUNIT/Database/Emotion/hap_neu/test_hap'
        #         self.max_samples = 1000

        self.batch_size = 1
        self.style_dim = 16

        self.Encoder_A = Encoder_A
        self.Encoder_B = Encoder_B
        self.Decoder_A = Decoder_A
        self.Decoder_B = Decoder_B
        self.discriminate_real = discriminate_real
        self.discriminate_fake = discriminate_fake

        self.gan_type = 'lsgan'

        self.gan_w = 10.0
        self.recon_x_w = 1.0
        self.recon_s_w = 1.0
        self.recon_c_w = 1.0
        self.recon_x_cyc_w = 10.0

        self.audio_len = 128  # = n_frames, time_length
        self.audio_ch = 24  # = num_mcep, num_features

        self.direction = 'A2B'

        self.model_name = 'EmoMUNIT'
        self.gan_type = 'lsgan'
        self.dataset_name = 'neu2surp'
        self.log_dir = 'logs'
        self.sample_dir = 'samples'
        self.checkpoint_dir = 'checkpoint'
        self.result_dir = 'results'

        self.sess = sess
        self.epoch = 1000
        self.iteration = 1000
        self.init_lr_D = 0.0001
        self.init_lr_G = 0.0002

        self.sample_freq = 1000  # 1000
        self.save_freq = 1000  # 1000

        self.sampling_rate = 16000
        self.frame_period = 5.0
        self.num_mcep = 24

    def build_model(self):
        self.lr_D = tf.placeholder(tf.float32, name='learning_rate_D')
        self.lr_G = tf.placeholder(tf.float32, name='learning_rate_G')

        # Iterate from train_data_A and train_data_A
        self.domain_A = tf.placeholder(tf.float32, shape=[self.batch_size, self.audio_ch, self.audio_len],
                                       name='domain_a')
        self.domain_B = tf.placeholder(tf.float32, shape=[self.batch_size, self.audio_ch, self.audio_len],
                                       name='domain_b')

        self.style_a = tf.placeholder(tf.float32, shape=[self.batch_size, 1, self.style_dim], name='style_a')
        self.style_b = tf.placeholder(tf.float32, shape=[self.batch_size, 1, self.style_dim], name='style_b')

        # encode
        content_a, style_a_prime = self.Encoder_A(self.domain_A)
        content_b, style_b_prime = self.Encoder_B(self.domain_B)

        # decode (within domain)
        x_aa = self.Decoder_A(content_B=content_a, style_A=style_a_prime)
        x_bb = self.Decoder_B(content_A=content_b, style_B=style_b_prime)

        # decode (cross domain)
        x_ba = self.Decoder_A(content_B=content_b, style_A=self.style_a, reuse=True)
        x_ab = self.Decoder_B(content_A=content_a, style_B=self.style_b, reuse=True)

        # encode again
        content_b_, style_a_ = self.Encoder_A(x_ba, reuse=True)
        content_a_, style_b_ = self.Encoder_B(x_ab, reuse=True)

        # decode again (if needed)
        if self.recon_x_cyc_w > 0:
            x_aba = self.Decoder_A(content_B=content_a_, style_A=style_a_prime, reuse=True)
            x_bab = self.Decoder_B(content_A=content_b_, style_B=style_b_prime, reuse=True)

            cyc_recon_A = L1_loss(x_aba, self.domain_A)
            cyc_recon_B = L1_loss(x_bab, self.domain_B)

        else:
            cyc_recon_A = 0.0
            cyc_recon_B = 0.0

        real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
        fake_A_logit, fake_B_logit = self.discriminate_fake(x_ba, x_ab)

        """ Define Loss """
        # Adversarial Loss
        G_ad_loss_a = generator_loss(self.gan_type, fake_A_logit)
        G_ad_loss_b = generator_loss(self.gan_type, fake_B_logit)

        # Discrimination Loss (real/fake)
        D_ad_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        D_ad_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        # Reconstruction Loss
        recon_A = L1_loss(x_aa, self.domain_A)  # reconstruction
        recon_B = L1_loss(x_bb, self.domain_B)  # reconstruction

        # Semi-CycleGAN Loss
        # For style, encourages diverse outputs given different style codes
        recon_style_A = L1_loss(style_a_, self.style_a)
        recon_style_B = L1_loss(style_b_, self.style_b)

        # For content, encourages the translated image to preserve semantic content of the input image
        recon_content_A = L1_loss(content_a_, content_a)
        recon_content_B = L1_loss(content_b_, content_b)

        # Attacker Loss
        Generator_A_loss = self.gan_w * G_ad_loss_a + \
                           self.recon_x_w * recon_A + \
                           self.recon_s_w * recon_style_A + \
                           self.recon_c_w * recon_content_A + \
                           self.recon_x_cyc_w * cyc_recon_A

        Generator_B_loss = self.gan_w * G_ad_loss_b + \
                           self.recon_x_w * recon_B + \
                           self.recon_s_w * recon_style_B + \
                           self.recon_c_w * recon_content_B + \
                           self.recon_x_cyc_w * cyc_recon_B

        # Defender Loss
        Discriminator_A_loss = self.gan_w * D_ad_loss_a
        Discriminator_B_loss = self.gan_w * D_ad_loss_b

        # Total Loss
        self.Generator_loss = Generator_A_loss + Generator_B_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training Variables """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'decoder' in var.name or 'encoder' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr_G, beta1=0.5, beta2=0.999).minimize(self.Generator_loss,
                                                                                          var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr_D, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss,
                                                                                          var_list=D_vars)

        """" Summary """
        self.all_G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.all_D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.R_A_loss = tf.summary.scalar("Reconstruction_A_loss", recon_A)
        self.R_B_loss = tf.summary.scalar("Reconstruction_B_loss", recon_B)
        self.recon_style_A = tf.summary.scalar("recon_style_A", recon_style_A)
        self.recon_style_B = tf.summary.scalar("recon_style_B", recon_style_B)
        self.recon_content_A = tf.summary.scalar("recon_content_A", recon_content_A)
        self.recon_content_B = tf.summary.scalar("recon_content_B", recon_content_B)

        self.G_A_loss = tf.summary.scalar("G_A_loss", Generator_A_loss)
        self.G_B_loss = tf.summary.scalar("G_B_loss", Generator_B_loss)
        self.G_ad_loss_a = tf.summary.scalar("G_ad_loss_a", G_ad_loss_a)
        self.G_ad_loss_b = tf.summary.scalar("G_ad_loss_b", G_ad_loss_b)

        self.D_A_loss = tf.summary.scalar("D_A_loss", Discriminator_A_loss)
        self.D_B_loss = tf.summary.scalar("D_B_loss", Discriminator_B_loss)

        self.G_loss = tf.summary.merge(
            [self.R_A_loss, self.R_B_loss, self.recon_style_A, self.recon_style_B, self.recon_content_A,
             self.recon_content_B, self.G_ad_loss_a, self.G_ad_loss_b, self.G_A_loss, self.G_B_loss, self.all_G_loss])
        self.D_loss = tf.summary.merge([self.D_A_loss, self.D_B_loss, self.all_D_loss])

        """ Speech: real and fake """
        self.real_A = self.domain_A
        self.real_B = self.domain_B

        self.fake_A = x_ba
        self.fake_B = x_ab

        """ Test Variables """
        self.test_domain_A = tf.placeholder(tf.float32, [1, self.audio_ch, None], name='test_domain_a')  # [1 24 None]
        self.test_domain_B = tf.placeholder(tf.float32, [1, self.audio_ch, None], name='test_domain_b')  # [1 24 None]

        self.test_style_a = tf.placeholder(tf.float32, [1, 1, self.style_dim], name='test_style_a')  # [1 1 16]
        self.test_style_b = tf.placeholder(tf.float32, [1, 1, self.style_dim], name='test_style_b')  # [1 1 16]

        test_content_a, test_style_a = self.Encoder_A(self.test_domain_A, reuse=True)
        test_content_b, test_style_b = self.Encoder_B(self.test_domain_B, reuse=True)

        self.test_fake_A = self.Decoder_A(content_B=test_content_b, style_A=self.test_style_a, reuse=True)
        self.test_fake_B = self.Decoder_B(content_A=test_content_a, style_B=self.test_style_b, reuse=True)

        self.test_recon_A = self.Decoder_A(content_B=test_content_a, style_A=test_style_a, reuse=True)
        self.test_recon_B = self.Decoder_B(content_A=test_content_b, style_B=test_style_b, reuse=True)

        """ Guided Speech Translation """
        self.content_audio = tf.placeholder(tf.float32, [1, self.audio_ch, self.audio_len], name='content_audio')
        self.style_audio = tf.placeholder(tf.float32, [1, self.audio_ch, self.audio_len], name='guide_style_audio_ch')

        if self.direction == 'A2B':
            guide_content_A, guide_style_A = self.Encoder_A(self.content_audio, reuse=True)
            guide_content_B, guide_style_B = self.Encoder_B(self.style_audio, reuse=True)

        else:
            guide_content_B, guide_style_B = self.Encoder_B(self.content_audio, reuse=True)
            guide_content_A, guide_style_A = self.Encoder_A(self.style_audio, reuse=True)

        self.guide_fake_A = self.Decoder_A(content_B=guide_content_B, style_A=guide_style_A, reuse=True)
        self.guide_fake_B = self.Decoder_B(content_A=guide_content_A, style_B=guide_style_B, reuse=True)

    def data_prepare(self, f0s_A, f0s_B, coded_sps_norm_A, coded_sps_norm_B):

        train_data_A = sample_train_data03(sps=list(coded_sps_norm_A), f0s=list(f0s_A), n_frames=self.audio_len)
        train_data_B = sample_train_data03(sps=list(coded_sps_norm_B), f0s=list(f0s_B), n_frames=self.audio_len)

        minlen = min(len(train_data_A), len(train_data_B))
        np.random.shuffle(train_data_A)
        np.random.shuffle(train_data_B)
        train_data_A = np.array(train_data_A[0:minlen])
        train_data_B = np.array(train_data_B[0:minlen])

        return train_data_A, train_data_B

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load FAILED...")

        # check sample_dir
        check_folder(self.sample_dir)

        '''Training loop for epoch'''

        # load data and extract features
        f0s_A, coded_sps_norm_A, log_f0s_mean_A, log_f0s_std_A, coded_sps_mean_A, coded_sps_std_A = vocoder_extract(
            self.train_A_dir)
        f0s_B, coded_sps_norm_B, log_f0s_mean_B, log_f0s_std_B, coded_sps_mean_B, coded_sps_std_B = vocoder_extract(
            self.train_B_dir)

        # load validation data
        wavs_val_A = load_wavs(wav_dir=self.validation_A_dir, sr=self.sampling_rate)
        # wavs_val_B = load_wavs(wav_dir=self.validation_B_dir, sr=self.sampling_rate)

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            train_data_A, train_data_B = self.data_prepare(f0s_A, f0s_B, coded_sps_norm_A, coded_sps_norm_B)
            print('Epoch[%d]: Input data sampled from %d A and %d B audio files: train_data_A' % (
            epoch, len(f0s_A), len(f0s_B)), np.shape(train_data_A), 'train_data_B', np.shape(train_data_B))

            lr_D, lr_G = self.init_lr_D * pow(0.995, epoch), self.init_lr_G * pow(0.995, epoch)
            for idx in range(start_batch_id, self.iteration):
                style_a = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, self.style_dim])
                style_b = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, 1, self.style_dim])

                idx_A = idx % len(train_data_A)
                idx_B = idx % len(train_data_B)
                domain_A = train_data_A[idx_A:idx_A + 1].astype('float32')
                domain_B = train_data_B[idx_B:idx_B + 1].astype('float32')

                train_feed_dict = {
                    self.style_a: style_a,
                    self.style_b: style_b,
                    self.lr_D: lr_D,
                    self.lr_G: lr_G,
                    self.domain_A: domain_A,
                    self.domain_B: domain_B
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss],
                                                       feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                batch_A_audios, batch_B_audios, fake_A, fake_B, _, g_loss, summary_str = \
                    self.sess.run([self.real_A, self.real_B, self.fake_A, self.fake_B, self.G_optim, \
                                   self.Generator_loss, self.G_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss), end='\r')

                # save generated samples
                if np.mod(counter + 1, self.sample_freq) == 0:
                    # A2B
                    idx_val_A = (counter // self.sample_freq) % len(wavs_val_A)
                    wav = wavs_val_A[idx_val_A]
                    wav = wav_padding(wav=wav, sr=self.sampling_rate, frame_period=self.frame_period, multiple=4)
                    # f0 conversion
                    f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=self.sampling_rate,
                                                           frame_period=self.frame_period)
                    f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                                    mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)
                    # sp normalization
                    coded_sp = world_encode_spectral_envelop(sp=sp, fs=self.sampling_rate, dim=self.num_mcep)
                    coded_sp_transposed = coded_sp.T
                    coded_sp_norm = (coded_sp_transposed - coded_sps_mean_A) / coded_sps_std_A
                    # random sampled style
                    test_style_b = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, self.style_dim])
                    # sp conversion (A2B)
                    coded_sp_converted_norm = self.sess.run(self.test_fake_B,
                                                            feed_dict={self.test_domain_A: np.array([coded_sp_norm]),
                                                                       self.test_style_b: test_style_b})
                    coded_sp_converted_norm_recon = self.sess.run(self.test_recon_A, feed_dict={
                        self.test_domain_A: np.array([coded_sp_norm])})
                    # [1,24,None]
                    # de-normalization
                    coded_sp_converted = coded_sp_converted_norm[0] * coded_sps_std_B + coded_sps_mean_B
                    coded_sp_converted = coded_sp_converted.T
                    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
                    coded_sp_converted_recon = coded_sp_converted_norm_recon[0] * coded_sps_std_A + coded_sps_mean_A
                    coded_sp_converted_recon = coded_sp_converted_recon.T
                    coded_sp_converted_recon = np.ascontiguousarray(coded_sp_converted_recon)
                    # combine converted f0, sp and ap
                    decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted,
                                                                         fs=self.sampling_rate)
                    decoded_sp_converted_recon = world_decode_spectral_envelop(coded_sp=coded_sp_converted_recon,
                                                                               fs=self.sampling_rate)
                    wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                             fs=self.sampling_rate, frame_period=self.frame_period)
                    wav_transformed_recon = world_speech_synthesis(f0=f0, decoded_sp=decoded_sp_converted_recon, ap=ap,
                                                                   fs=self.sampling_rate,
                                                                   frame_period=self.frame_period)
                    # write .wav file
                    path_A2B = '{}/fake_A2B_id{:03d}_iter{:03d}K.wav'.format(self.sample_dir, idx_val_A,
                                                                             counter // 1000)
                    path_A2A = '{}/recon_A2A_id{:03d}_iter{:03d}K.wav'.format(self.sample_dir, idx_val_A,
                                                                              counter // 1000)
                    print(self.sampling_rate)
                    save_audio(wav=wav_transformed, path=path_A2B, sr=self.sampling_rate)
                    save_audio(wav=wav_transformed_recon, path=path_A2A, sr=self.sampling_rate)

                # save checkpoints
                if np.mod(counter + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id reset to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model for final step
            self.save(self.checkpoint_dir, counter)

        print(" [*] Training finished!")

    def test(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # load check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load FAILED...")

        # check result_dir
        check_folder(self.result_dir)

        # write html for visual comparison

        # Get statistic from train_A, train_B
        _, _, log_f0s_mean_A, log_f0s_std_A, coded_sps_A_mean, coded_sps_A_std = vocoder_extract(self.train_A_dir)
        _, _, log_f0s_mean_B, log_f0s_std_B, coded_sps_B_mean, coded_sps_B_std = vocoder_extract(self.train_B_dir)
        print('std_log_src:', log_f0s_std_A, 'std_log_target', log_f0s_std_B)

        # A2B
        test_files_A = os.listdir(self.validation_A_dir)
        for i in range(len(test_files_A)):
            file = test_files_A[i]
            filepath = os.path.join(self.validation_A_dir, file)
            wav, _ = librosa.load(filepath, sr=self.sampling_rate, mono=True)
            wav = wav_padding(wav=wav, sr=self.sampling_rate, frame_period=self.frame_period, multiple=4)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=self.sampling_rate, frame_period=self.frame_period)

            # f0 conversion
            f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A,
                                            mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

            # sp normalization
            coded_sp = world_encode_spectral_envelop(sp=sp, fs=self.sampling_rate, dim=self.num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed - coded_sps_A_mean) / coded_sps_A_std

            # random sampled style
            test_style_b = np.random.normal(loc=0.0, scale=1.0, size=[1, 1, self.style_dim])

            # sp conversion (A2B)
            coded_sp_converted_norm = self.sess.run(self.test_fake_B,
                                                    feed_dict={self.test_domain_A: np.array([coded_sp_norm]),
                                                               self.test_style_b: test_style_b})
            # [1,24,None]

            # print('coded_sp_converted_norm', np.shape(coded_sp_converted_norm[0]), 'coded_sps_B_mean', np.shape(coded_sps_B_mean), 'coded_sps_B_std:', np.shape(coded_sps_B_std))
            coded_sp_converted = coded_sp_converted_norm[0] * coded_sps_B_std + coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, fs=self.sampling_rate)
            wav_transformed = world_speech_synthesis(f0=f0_converted, decoded_sp=decoded_sp_converted, ap=ap,
                                                     fs=self.sampling_rate, frame_period=self.frame_period)
            # librosa.output.write_wav(os.path.join(self.result_dir, os.path.basename(file)), wav_transformed, self.sampling_rate)
            sf.write(os.path.join(self.result_dir, os.path.basename(file)), wav_transformed, self.sampling_rate)
            print('converting test samples: [%d/%d]' % (i + 1, len(test_files_A)), end='\r')

        print(" [*] Testing finished!")

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.gan_type)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)




with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    gan = EmoMUNIT(sess)
    gan.build_model()
    gan.test()
