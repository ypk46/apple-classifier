import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers, optimizers
from keras.initializers import glorot_uniform, Constant
from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization, concatenate
from keras.preprocessing.image import ImageDataGenerator

nClasses = 2
input_shape = (224, 224, 3)

# Kernel Initializers
xavier = glorot_uniform(seed=None)

# Bias Initializers
constant_bias = Constant(0.2)

# Kernel Regularizer
k_reg = regularizers.l2(l=0.0002)


def inception(layer, filters_batch):

    # First Tower
    first_tower = Conv2D(filters=filters_batch[0], kernel_size=1,
                         padding='same', activation='relu',
                         kernel_initializer=xavier,
                         bias_initializer=constant_bias,
                         kernel_regularizer=k_reg)(layer)

    # Second Tower
    second_tower = Conv2D(filters=filters_batch[1], kernel_size=1,
                          padding='same', activation='relu',
                          kernel_initializer=xavier,
                          bias_initializer=constant_bias,
                          kernel_regularizer=k_reg)(layer)

    second_tower = Conv2D(filters=filters_batch[2], kernel_size=3,
                          activation='relu', padding='same',
                          kernel_initializer=xavier,
                          bias_initializer=constant_bias,
                          kernel_regularizer=k_reg)(second_tower)

    # Third Tower
    third_tower = Conv2D(filters=filters_batch[3], kernel_size=1,
                         padding='same', activation='relu',
                         kernel_initializer=xavier,
                         bias_initializer=constant_bias,
                         kernel_regularizer=k_reg)(layer)

    third_tower = Conv2D(filters=filters_batch[4], kernel_size=5,
                         activation='relu', padding='same',
                         kernel_initializer=xavier,
                         bias_initializer=constant_bias,
                         kernel_regularizer=k_reg)(layer)

    # Fourth Tower
    fourth_tower = MaxPooling2D(pool_size=3, strides=1, padding='same')(layer)

    fourth_tower = Conv2D(filters=filters_batch[5], kernel_size=1,
                          padding='same', activation='relu',
                          kernel_initializer=xavier,
                          bias_initializer=constant_bias,
                          kernel_regularizer=k_reg)(fourth_tower)

    layer = concatenate(
        [first_tower, second_tower, third_tower, fourth_tower], axis=3)

    return layer


def aux_classifier(layer):
    layer = AveragePooling2D(pool_size=3, strides=2, padding='same')(layer)
    layer = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid',
                   activation='relu', kernel_regularizer=k_reg)(layer)
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu',
                  kernel_regularizer=k_reg)(layer)
    layer = Dropout(0.7)(layer)
    layer = Dense(nClasses, activation='softmax',
                  kernel_regularizer=k_reg)(layer)

    return layer


def createModel(inputShape):

    # Input
    input = Input(inputShape)

    conv1 = Conv2D(filters=64, kernel_size=7, strides=2,
                   padding='same', activation='relu',
                   kernel_initializer=xavier,
                   bias_initializer=constant_bias,
                   kernel_regularizer=k_reg)(input)

    pool1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(conv1)

    norm1 = BatchNormalization(axis=3)(pool1)

    conv2 = Conv2D(filters=64, kernel_size=1,
                   activation="relu",
                   kernel_initializer=xavier,
                   kernel_regularizer=k_reg,
                   bias_initializer=constant_bias)(norm1)

    conv3 = Conv2D(filters=192, kernel_size=3,
                   activation='relu', kernel_initializer=xavier,
                   kernel_regularizer=k_reg,
                   bias_initializer=constant_bias)(conv2)

    norm2 = BatchNormalization(axis=3)(conv3)

    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same')(norm2)

    inception1 = inception(pool2, filters_batch=[64, 96, 128, 16, 32, 32])

    inception2 = inception(inception1, filters_batch=[
                           128, 128, 192, 32, 96, 64])

    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same')(inception2)

    inception3 = inception(pool3, filters_batch=[192, 96, 208, 16, 48, 64])

    aux1 = aux_classifier(inception3)

    inception4 = inception(inception3, filters_batch=[
                           160, 112, 224, 24, 64, 64])
    inception5 = inception(inception4, filters_batch=[
                           128, 128, 256, 24, 64, 64])
    inception6 = inception(inception5, filters_batch=[
                           112, 144, 288, 32, 64, 64])
    aux2 = aux_classifier(inception6)
    inception7 = inception(inception6, filters_batch=[
                           256, 160, 320, 32, 128, 128])

    pool4 = MaxPooling2D(pool_size=3, strides=2)(inception7)

    inception8 = inception(pool4, filters_batch=[256, 160, 320, 32, 128, 128])
    inception9 = inception(inception8, filters_batch=[
                           384, 192, 384, 48, 128, 128])
    avg = AveragePooling2D(pool_size=4, strides=1)(inception9)

    flat = Flatten()(avg)
    drop = Dropout(0.4)(flat)
    output = Dense(nClasses, activation='softmax',
                   kernel_regularizer=k_reg)(drop)

    model = Model(inputs=input, outputs=[output])
    return model


# Data Preparation
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

optimizer = optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

leave_model = createModel(input_shape)

leave_model.compile(optimizer=optimizer,
                    loss="categorical_crossentropy", metrics=["accuracy"], loss_weights=[1.])


# for epoch in range(100):

training_set = datagen.flow_from_directory(
    'dataset/training',
    target_size=(224, 224),
    batch_size=100)

test_set = datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=100)

results = leave_model.fit_generator(
    training_set,
    steps_per_epoch=8856,
    epochs=15,
    validation_data=test_set,
    validation_steps=2216)

# leave_model.save("mango_model_" + epoch + ".h5")

leave_model.summary()

leave_model.save("mango_model.h5")
