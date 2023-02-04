import keras
import tensorflow as tf
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import os, time
from keras.regularizers import l1, l2
import keras.backend as K
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
    Activation, ZeroPadding2D
from keras.layers import add, Flatten
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
import keras
from keras.activations import sigmoid


def plain_cnn_256ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1))

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_256ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_CBAM_256ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="relu", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def plain_cnn_128ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1))

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_128ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_CBAM_128ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="relu", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def plain_cnn_64ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2))

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_64ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="sigmoid", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


def resnet_CBAM_64ms(input_layer, nb_classes):
    x = input_layer

    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(1, 2), use_shortcut=True, with_conv_shortcut=True)
    x = CbamBlock2D().layer(x)

    x = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)

    x = keras.layers.Dropout(0.5)(x)
    regularSet = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    x = keras.layers.Dense(8, activation="relu", kernel_regularizer=regularSet)(x)

    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=x)
    return model


class CbamBlock2D(object):
    def __init__(self, reduction=8):
        self.reduction = reduction

    def layer(self, inputLayer):
        inputLayer = self.channel_attention(inputLayer)
        inputLayer = self.spacial_attention(inputLayer)

        return inputLayer

    def channel_attention(self, inputLayer):
        channels = inputLayer.shape.as_list()[-1]
        avg_x = GlobalAveragePooling2D()(inputLayer)
        avg_x = Reshape((1, 1, channels))(avg_x)

        max_x = GlobalMaxPooling2D()(inputLayer)
        max_x = Reshape((1, 1, channels))(max_x)

        Dense_One = Dense(units=int(channels) // self.reduction, activation='relu',
                          kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        Dense_Two = Dense(units=int(channels), activation='relu', kernel_initializer='he_normal',
                          use_bias=True, bias_initializer='zeros')

        avg_x = Dense_One(avg_x)
        avg_x = Dense_Two(avg_x)

        max_x = Dense_One(max_x)
        max_x = Dense_Two(max_x)

        cbam_feature = Add()([avg_x, max_x])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        return multiply([inputLayer, cbam_feature])

    def spacial_attention(self, input_feature):
        kernel_size = 7

        cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        cbam_feature = keras.layers.Conv2D(filters=1,
                                           kernel_size=kernel_size,
                                           strides=1,
                                           padding='same',
                                           activation='sigmoid',
                                           use_bias=False, kernel_initializer='he_normal')(concat)

        return multiply([input_feature, cbam_feature])


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None, active=True):
    normal_func = InstanceNormalization

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer="he_normal")(x)
    x = normal_func()(x)
    if active:
        x = Activation('relu')(x)

    #     x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    #     x = keras.layers.Dropout(0.2)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), use_shortcut=False, with_conv_shortcut=False):
    if use_shortcut:
        active = False
    else:
        active = True

    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same', active=active)

    if use_shortcut:
        if with_conv_shortcut:
            shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size, active=active)
            x = add([x, shortcut])
            x = Activation('relu')(x)
            return x
        else:
            x = add([x, inpt])
            x = Activation('relu')(x)
            return x
    else:
        return x


def printMetrics(real, predict):
    TP = np.where((predict == 1) & (real == 1))[0].size
    FP = np.where((predict == 1) & (real == 0))[0].size
    FN = np.where((predict == 0) & (real == 1))[0].size
    TN = np.where((predict == 0) & (real == 0))[0].size

    accu = round((TP + TN) * 100 / real.size, 6)
    precision = round(TP * 100 / (TP + FP), 6)
    recall = round(TP * 100 / (TP + FN), 6)

    f1_score = round(2 * (precision * recall) / (precision + recall), 4)
    print(f"accu:{accu}%；precision:{precision}%；recall:{recall}%；f1_score:{f1_score}%；")


def train_model(model, train_x, train_y, val_x, val_y, trainEpochs, trainBatchSize,
                verbose=2, outDir="/input/out/", binSize="64ms", modelName="cnn"):
    from keras.callbacks import Callback
    class LossHistory(Callback):

        def on_train_begin(self, logs={}):
            self.maxLoss = 0
            self.maxAccu = 0
            print('begin train')

        def on_epoch_end(self, epoch, logs={}):
            presentLoss = logs.get("val_loss")
            presentAccu = logs.get("val_acc")

            if presentAccu > self.maxAccu:
                self.maxLoss = round(presentLoss, 4)
                self.maxAccu = round(presentAccu, 4)

                print("update max_loss:%s ;max_accu:%s" % (self.maxLoss, self.maxAccu))
            else:
                print("max_loss:%s ;max_accu:%s" % (self.maxLoss, self.maxAccu))

        def on_train_end(self, logs={}):
            print("max_loss:%s ;max_accu:%s" % (self.maxLoss, self.maxAccu))

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.00005,
                                                  verbose=1)

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=40, verbose=2)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    saveBestModel = keras.callbacks.ModelCheckpoint(
        f"{outDir}time_{binSize}_{modelName}_" + "{epoch:03d}_{val_acc:.4f}_{val_loss:.4f}.h5",
        monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
    csvLogger = keras.callbacks.CSVLogger(outDir + "train.log", separator=',', append=False)

    callbacks = [reduce_lr, early_stopping, saveBestModel, csvLogger, LossHistory()]

    history = model.fit(train_x, train_y, epochs=trainEpochs, batch_size=trainBatchSize, verbose=verbose,
                        validation_data=(val_x, val_y), callbacks=callbacks)

    history_dict = history.history
