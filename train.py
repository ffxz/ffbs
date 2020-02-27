
import os
import numpy as np

import tensorflow as tf
import keras
from keras import layers
import numpy as np
from tensorflow.python.platform import flags

import dataset

import os
# https://sklearn.org/modules/preprocessing.html
from sklearn import preprocessing

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

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

flags.DEFINE_string('task', 'libri', 'set task name of this program')
flags.DEFINE_string('train_dataset', 'train-clean-100', 'set the training dataset')
flags.DEFINE_string('mode', 'train', 'choice =[train, test, infer]')
flags.DEFINE_string('feature_mode', 'magnitude', 'choice =[magnitude, complex]')
flags.DEFINE_string('workspace', '/home/dataset', 'set dir to your root')#/home6/libin/denoise/BinNew
flags.DEFINE_string('speech_dir', 'data/clean', 'set dir to your speech')
flags.DEFINE_string('noisy_dir', 'data/noisy', 'setest dir to your noise')
flags.DEFINE_integer('snr', 5, 'snr')
flags.DEFINE_integer('n_hop', 1, 'frame hop length')
flags.DEFINE_float('lr', 0.01, 'learning rate')
flags.DEFINE_integer('fs', 16000, 'fs')
flags.DEFINE_integer('n_concat', 8, 'frame length')
flags.DEFINE_integer('magnification', 1, 'feature type')
flags.DEFINE_integer('sample_rate', 16000, 'sample_rate')
flags.DEFINE_integer('n_window', 512, 'n_window')#512
flags.DEFINE_integer('n_overlap', 256, 'n_overlap')#256
flags.DEFINE_integer('batch_size', 6, 'batch_size')#6


FLAGS = flags.FLAGS
log10_fac = 1 / np.log(10)
K.set_epsilon(1e-08)

def train():
    FLAGS.n_window = 512
    FLAGS.n_overlap = 256
    args = FLAGS
    FLAGS.workspace = '/home3/sining/ffbs/'#'/home2/libin/yan2/denoise/BinNew_218/'
    n_freq = args.n_window // 2 + 1
    model = B2_power(n_freq)
    checkpoint_path = os.path.join(FLAGS.workspace, 'log/uu_006.ckpt')
    if not os.path.exists(os.path.split(checkpoint_path)[0]):
        os.mkdir(os.path.split(checkpoint_path)[0])
    checkpoint_h5 = os.path.join(FLAGS.workspace, 'log/uu_006.h5')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_h5, save_weights_only=True, verbose=1)
    # 提早结束训练   'restore_best_weights'选择是否要保存最好的模型参数
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5, verbose=1, mode='auto')
    # 如果需要学习率需要动态更新需要使用下面这种形式的优化器，如果不要更新直接使用上面这种优化器形式就可以
    opt = keras.optimizers.Adam(lr=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 如果几个epoch loss不下降，就减小学习率
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=0.000001,
                                                  mode='auto', verbose=1, epsilon=0.001)
    #model.compile(optimizer=opt,
    #              loss={'bin_out_1': 'mse'})

    model.compile(loss={'bin_out_1': 'categorical_crossentropy'},
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    model.summary()
    speech_train = '/home3/sining/sound_class/train/'

    speech_dev = '/home3/sining/sound_class/dev/'

    model.fit_generator(
        dataset.get_gen_train(FLAGS, speech_train, batch_size=FLAGS.batch_size),
        epochs=80, verbose=1, steps_per_epoch=1300 // FLAGS.batch_size,
        validation_data=dataset.get_gen_train(FLAGS, speech_dev, batch_size=FLAGS.batch_size),
        validation_steps=300 // FLAGS.batch_size,
        callbacks=[reduce_lr, cp_callback, keras.callbacks.TensorBoard(log_dir=checkpoint_dir),
                   earlyStopping], initial_epoch=0)
    model.save_weights(checkpoint_h5)
    del model


def train_mfcc():
    FLAGS.n_window = 512
    FLAGS.n_overlap = 256
    args = FLAGS
    FLAGS.workspace = '/home3/sining/ffbs/'
    n_freq = 39
    model = Mfcc_Net(n_freq)
    checkpoint_path = os.path.join(FLAGS.workspace, 'log/uu_mfcc.ckpt')
    if not os.path.exists(os.path.split(checkpoint_path)[0]):
        os.mkdir(os.path.split(checkpoint_path)[0])
    checkpoint_h5 = os.path.join(FLAGS.workspace, 'log/uu_mfcc.h5')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_h5, save_weights_only=True, verbose=1)
    # 提早结束训练   'restore_best_weights'选择是否要保存最好的模型参数
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5, verbose=1, mode='auto')
    # 如果需要学习率需要动态更新需要使用下面这种形式的优化器，如果不要更新直接使用上面这种优化器形式就可以
    opt = keras.optimizers.Adam(lr=FLAGS.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # 如果几个epoch loss不下降，就减小学习率
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=0.000001,
                                                  mode='auto', verbose=1, epsilon=0.001)

    model.compile(loss={'bin_out_1': 'categorical_crossentropy'},
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    model.summary()
    speech_train = '/home3/sining/sound_class/train/'

    speech_dev = '/home3/sining/sound_class/dev/'

    model.fit_generator(
        dataset.get_gen_mfcc_train(FLAGS, speech_train, batch_size=FLAGS.batch_size),
        epochs=80, verbose=1, steps_per_epoch=1300 // FLAGS.batch_size,
        validation_data=dataset.get_gen_mfcc_train(FLAGS, speech_dev, batch_size=FLAGS.batch_size),
        validation_steps=300 // FLAGS.batch_size,
        callbacks=[reduce_lr, cp_callback, keras.callbacks.TensorBoard(log_dir=checkpoint_dir),
                   earlyStopping], initial_epoch=0)
    model.save_weights(checkpoint_h5)
    del model


if __name__ == '__main__':
    train_mfcc()
    #dataset.ff(FLAGS, '/home3/sining/sound_class/train/')
