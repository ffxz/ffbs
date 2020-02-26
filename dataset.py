import os
import numpy as np
import random

import soundfile
import librosa
import keras

from scipy import signal



def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def calc_sp_by(args, audio, mode):
    n_window = args.n_window
    n_overlap = args.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
        audio,
        window=ham_win,
        nperseg=n_window,
        noverlap=n_overlap,
        detrend=False,
        return_onesided=True,
        mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x

def log_sp(x):
    return np.log(x + 1e-08)


def Write_CNN_by(FLAGS, speech_audio, wav_name, classes):

    speech_x = calc_sp_by(FLAGS, speech_audio, mode='magnitude')
    clean_x = speech_x
    clean_all = log_sp(clean_x).astype(np.float32)
    #clean_all = clean_all[:, :, np.newaxis]
    label = wav_name.split('.')[0].split('-')[-1]
    label = keras.utils.to_categorical(label, classes)

    return clean_all, label

def write_mfcc(speech_audio, wav_name, classes):

    mfcc = librosa.feature.mfcc(speech_audio, 16000, n_mfcc=39)
    label = wav_name.split('.')[0].split('-')[-1]
    label = keras.utils.to_categorical(label, classes)
    return mfcc, label


def get_gen_train(args, speech_dir, batch_size):
    workspace = args.workspace
    # 这里输入的干净语音和带噪语音的文件名是一样的
    # shuffle
    speech_names = os.listdir(speech_dir)
    speech_names.sort()

    c = random.sample(range(1, 100000), 1)
    c = c[0]
    np.random.seed(c)
    index = [ii for ii in range(len(speech_names))]
    np.random.shuffle(index)
    speech_names = np.array(speech_names)
    speech_names = speech_names[index]
    while 1:
        for i in range(0, len(speech_names) - batch_size, batch_size):
            xx = []
            yy = []
            for speech_na in range(i, i + batch_size):
                speech_audio = read_audio(os.path.join(speech_dir, speech_names[speech_na]), 16000)
                s_audio = speech_audio[0]
                #while len(s_audio) < 513:
                #    idx_s_na = random.sample(range(len(speech_names)), 1)
                #    idx_s_na = idx_s_na[0]
                #    speech_audio = read_audio(os.path.join(speech_dir, speech_names[idx_s_na]), 16000)
                #    s_audio = speech_audio[0]
                x_all, label_all = Write_CNN_by(args, s_audio, speech_names[speech_na], 50)

                xx.append(x_all)
                yy.append(label_all)
            #the dypte is very very important
            #xx = keras.preprocessing.sequence.pad_sequences(xx, maxlen=300, dtype='float32', padding='pre', truncating='pre', value=0.0)
            #yy = keras.preprocessing.sequence.pad_sequences(yy, maxlen=50, dtype='float32', padding='pre', truncating='pre', value=0.0)
            xx = np.array(xx)
            yy = np.array(yy)

            yield (xx, yy)

def ff(args,speech_dir):
    speech_names = os.listdir(speech_dir)
    #speech_names = speech_names[0:15]
    a = []
    for item in speech_names:
        speech_audio = read_audio(os.path.join(speech_dir, item), 16000)[0]
    #speech_audio = read_audio('/home3/sining/sound_class/train/4-183487-A-1.wav', 16000)[0]

        #x_all, label_all = Write_CNN_by(args, speech_audio, '4-183487-A-1.wav', 50)
        x_all, label_all = Write_CNN_by(args, speech_audio, item, 50)
        if len(x_all)!=311:
            print(item)

        a.append(x_all)
    print('fdf')