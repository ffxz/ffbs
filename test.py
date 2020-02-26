

from dataset import read_audio



import numpy as np

import tensorflow as tf
import keras
from keras import layers
import numpy as np
from tensorflow.python.platform import flags

import dataset

import os


import time

from model import *
import librosa
from keras.utils import multi_gpu_model
#from tensorflow.keras.callbacks import ReduceLROnPlateau
#from tensorflow.keras.utils import plot_model

from keras.models import model_from_json

from keras import backend as K
import keras.backend.tensorflow_backend as KTF
K.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

flags.DEFINE_string('task', 'libri', 'set task name of this program')
flags.DEFINE_string('train_dataset', 'train-clean-100', 'set the training dataset')
flags.DEFINE_string('mode', 'test', 'choice =[train, test, infer]')
flags.DEFINE_string('feature_mode', 'magnitude', 'choice =[magnitude, complex]')
flags.DEFINE_string('workspace', '/home/dataset', 'set dir to your root')#/home6/libin/denoise/BinNew
flags.DEFINE_string('speech_dir', 'data/clean', 'set dir to your speech')
flags.DEFINE_string('noisy_dir', 'data/noisy', 'setest dir to your noise')
flags.DEFINE_integer('snr', 5, 'snr')
flags.DEFINE_integer('n_hop', 1, 'frame hop length')
flags.DEFINE_float('lr', 0.0006, 'learning rate')
flags.DEFINE_integer('fs', 16000, 'fs')
flags.DEFINE_integer('n_concat', 8, 'frame length')
flags.DEFINE_integer('magnification', 1, 'feature type')
flags.DEFINE_integer('sample_rate', 16000, 'sample_rate')
flags.DEFINE_integer('n_window', 512, 'n_window')#512
flags.DEFINE_integer('n_overlap', 256, 'n_overlap')#256
flags.DEFINE_integer('batch_size', 6, 'batch_size')#6

flags.DEFINE_float('te_snr', 5, 'te_snr')
flags.DEFINE_float('tr_snr', 5, 'tr_snr')

FLAGS = flags.FLAGS
log10_fac = 1 / np.log(10)
K.set_epsilon(1e-08)





def test_long_b2_512_448():
    workspace = FLAGS.workspace
    speech_dir = '/home/dataset/Corpus/NSDTSEA/clean/test/'

    FLAGS.n_window = 512
    FLAGS.n_overlap = 448
    FLAGS.workspace = '/home2/libin/yan2/denoise/BinNew_218'
    workspace = FLAGS.workspace
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]

    speech_names.sort()

    n_concat = FLAGS.n_concat
    n_freq = FLAGS.n_window // 2 + 1
    model = B2_power(n_freq, True)
    checkpoint_path = os.path.join(FLAGS.workspace, 'log_noise/log_B2_power_512_448_noise/uu.ckpt')
    checkpoint_h5 = os.path.join(FLAGS.workspace, 'log_noise/log_B2_power_512_448_noise/uu.h5')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    opt = keras.optimizers.Adam(lr=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model = models.load_model(checkpoint_path)
    model.load_weights(checkpoint_h5)
    model.compile(optimizer=opt,
                  loss={'bin_out_1': 'mse'})
    #model = multi_gpu_model(model, gpus=2)
    for everyspeech in speech_names:
        (tr_x, tr_y) = dataset.get_long_seq_test(FLAGS, speech_dir, [everyspeech])

        pred = model.predict(tr_x)







