

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
    dense2 = Dropout(0.5)(dense2)
    classes = 50

    predict = Dense(classes, activation='softmax', name='bin_out_1')(dense2)


    print('point')
    return keras.Model(inputs=inputs, outputs=predict)





def B3_power(n_freq, trainable):
    #[B, T, 257, 1]
    inputs = keras.Input(shape=(None, n_freq, 1))  # Returns a placeholder tensor
    inputs2 = inputs
    # [B, T, 257, 1]
    lay1 = layers.Conv2D(16, kernel_size=(1, 3), strides=(1, 2), padding='VALID',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs2)
    #[128, 16]
    layer1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(lay1)
    #layer1 = keras.layers.BatchNormalization(trainable=trainable)(layer1)
    layer1 = keras.layers.Activation('elu')(layer1)

    #a1 [batch,t,64,32]
    layer2 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer1)
    #layer2 = keras.layers.BatchNormalization(trainable=trainable)(layer2)
    layer2 = keras.layers.Activation('elu')(layer2)

    #a2 [batch,t,32,64]
    layer3 = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer2)
    #layer3 = keras.layers.BatchNormalization(trainable=trainable)(layer3)
    layer3 = keras.layers.Activation('elu')(layer3)
    #a3 [batch,t,16,64]
    layer4 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer3)
    #layer4 = keras.layers.BatchNormalization(trainable=trainable)(layer4)
    layer4 = keras.layers.Activation('elu')(layer4)

    #a4 [batch,t,8,64]
    layer5 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer4)
    #layer5 = keras.layers.BatchNormalization(trainable=trainable)(layer5)
    layer5 = keras.layers.Activation('elu')(layer5)
    # a5 [batch,t,4,64]
    layer6 = layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                           kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                           bias_initializer=tf.keras.initializers.Constant(value=0.1),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer5)
    #layer6 = keras.layers.BatchNormalization(trainable=trainable)(layer6)
    layer6 = keras.layers.Activation('elu')(layer6)

    #[batch,t,2,64]
    layer7 = layers.Conv2DTranspose(512, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer6)
    # layer7 = keras.layers.BatchNormalization(trainable=trainable)(layer7)
    layer7 = keras.layers.Activation('elu')(layer7)
    layer7 = keras.layers.Concatenate()([layer7, layer5])
    #layer7 = keras.layers.Add()([layer7, layer5])
    # b3 [batch,t,4,88]
    layer8 = layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer7)
    # layer8 = keras.layers.BatchNormalization(trainable=trainable)(layer8)
    layer8 = keras.layers.Activation('elu')(layer8)
    layer8 = keras.layers.Concatenate()([layer8, layer4])
    # b2 [batch,t,8,88]
    layer9 = layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                    bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer8)
    # layer9 = keras.layers.BatchNormalization(trainable=trainable)(layer9)
    layer9 = keras.layers.Activation('elu')(layer9)
    layer9 = keras.layers.Concatenate()([layer9, layer3])
    # b1 [batch,t,16,88]
    layer10 = layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                     bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer9)
    # layer10 = keras.layers.BatchNormalization(trainable=trainable)(layer10)
    layer10 = keras.layers.Activation('elu')(layer10)
    layer10 = keras.layers.Concatenate()([layer10, layer2])
    # b0 [batch,t,32,88]
    layer11 = layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                     bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer10)
    # layer11 = keras.layers.BatchNormalization(trainable=trainable)(layer11)
    layer11 = keras.layers.Activation('elu')(layer11)
    layer11 = keras.layers.Concatenate()([layer11, layer1])

    # end [batch,t,64,88]
    layer12 = layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=(1, 2), padding='SAME',
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                     bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer11)
    # layer12 = keras.layers.BatchNormalization(trainable=trainable)(layer12)
    layer12 = keras.layers.Activation('elu')(layer12)
    layer12 = keras.layers.Concatenate()([layer12, lay1])
    #[batch, t, 128, 24]

    layer13 = layers.Conv2DTranspose(1, kernel_size=(1, 3), strides=(1, 2), padding='VALID',
                                     kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                     bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(layer12)
    layer13 = keras.layers.Activation('relu')(layer13)
    #[257, 1]



    #layer12 = keras.layers.Activation('sigmoid')(layer12)
    layer13 = keras.layers.Subtract(name='bin_out_1')([inputs2, layer13])
    print('point')
    return keras.Model(inputs=inputs, outputs=layer13)




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

