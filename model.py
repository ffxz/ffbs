

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.models import Model
import keras
from keras import layers

def keras_batchnormalization_relu(layer):
    BN = BatchNormalization()(layer)
    ac = PReLU()(BN)
    return ac

def B2_power(n_freq):
    #[B, T, 257]
    inputs = keras.Input(shape=(311, n_freq))  # Returns a placeholder tensor
    inputs2 = inputs
    # [B, T, 257]
    lay1 = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs2)

    layer1 = layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(lay1)
    layer1 = keras.layers.Activation('elu')(layer1)

    layer2 = layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer1)
    layer2 = keras.layers.Activation('elu')(layer2)

    layer3 = layers.Conv1D(filters=16, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer2)
    layer3 = keras.layers.Activation('elu')(layer3)

    layer4 = layers.Conv1D(filters=8, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer3)
    layer4 = keras.layers.Activation('elu')(layer4)

    layer4 = Flatten()(layer4)

    dense2 = Dense(1024, activation='relu')(layer4)
    #dense2 = Dropout(0.5)(dense2)
    classes = 50

    predict = Dense(classes, activation='softmax', name='bin_out_1')(dense2)


    print('point')
    return keras.Model(inputs=inputs, outputs=predict)


def Mfcc_Net(n_freq):
    #[B, T, 257]
    inputs = keras.Input(shape=(157, n_freq))  # Returns a placeholder tensor
    inputs2 = inputs
    # [B, T, 257]
    lay1 = layers.Conv1D(filters=78, kernel_size=3, strides=1, padding='VALID',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs2)

    layer1 = layers.Conv1D(filters=39, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(lay1)
    layer1 = keras.layers.Activation('elu')(layer1)

    layer2 = layers.Conv1D(filters=18, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer1)
    layer2 = keras.layers.Activation('elu')(layer2)

    layer3 = layers.Conv1D(filters=18, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer2)
    layer3 = keras.layers.Activation('elu')(layer3)

    layer4 = layers.Conv1D(filters=18, kernel_size=3, strides=2, padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer3)
    layer4 = keras.layers.Activation('elu')(layer4)

    layer4 = Flatten()(layer4)

    dense2 = Dense(1024, activation='relu')(layer4)
    #dense2 = Dropout(0.5)(dense2)
    classes = 50

    predict = Dense(classes, activation='softmax', name='bin_out_1')(dense2)


    print('point')
    return keras.Model(inputs=inputs, outputs=predict)





def Gru_net(n_freq):
    inputs = keras.Input((311, n_freq))  # Returns a placeholder tensor
    inputs2 = inputs
    layer1 = inputs2

    layer2 = keras.layers.GRU(units=128, activation='elu', return_sequences=True)(layer1)
    layer3 = keras.layers.GRU(units=128, activation='elu', return_sequences=True)(layer2)
    layer4 = keras.layers.GRU(units=128, activation='elu', return_sequences=True)(layer3)
    layer5 = keras.layers.GRU(units=128, activation='elu', return_sequences=True)(layer4)

    layer5 = Flatten()(layer5)

    dense2 = Dense(1024, activation='relu')(layer5)
    dense2 = Dropout(0.5)(dense2)

    classes = 50

    predict = Dense(classes, activation='softmax', name='bin_out_1')(dense2)



    return keras.Model(inputs=inputs, outputs=predict)




def AlexNet2(inputs, classes=50, prob=0.5):
 
    conv1 = Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid')(inputs)
    conv1 = keras_batchnormalization_relu(conv1)

    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(conv1)
 
    conv2 = Conv2D(filters=256, kernel_size=(5, 5), padding='same')(pool1)
    conv2 = keras_batchnormalization_relu(conv2)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(conv2)
 
    conv3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(pool2)
    conv3 = PReLU()(conv3)
 
    conv4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same')(conv3)
    conv4 = PReLU()(conv4)
 
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(conv4)
    conv5 = PReLU()(conv5)
 
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(conv5)
 
    dense1 = Flatten()(pool3)
    dense1 = Dense(4096, activation='relu')(dense1)
    dense1 = Dropout(prob)(dense1)
 
    dense2 = Dense(4096, activation='relu')(dense1)
    dense2 = Dropout(prob)(dense2)
 
    predict = Dense(classes, activation='softmax')(dense2)
 
    model = Model(inputs=inputs, outputs=predict)
    return model

