import numpy as np
import sys
import matplotlib.pyplot as plt
from keras import regularizers, optimizers
from keras.initializers import glorot_uniform, Constant
from keras.models import Model
from keras.layers import Dense, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Flatten, Input, BatchNormalization, concatenate, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator

n = 0
nClasses = 4
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

    pad1 = ZeroPadding2D(padding=2)(input)
    conv1 = Conv2D(filters=96, kernel_size=9,
                   strides=4, activation='relu')(pad1)

    pool1 = MaxPooling2D(pool_size=3, strides=2)(conv1)

    norm1 = BatchNormalization(axis=3)(pool1)

    conv2 = Conv2D(filters=256, kernel_size=5, padding='same',
                   activation="relu")(norm1)

    norm2 = BatchNormalization(axis=3)(conv2)

    pool2 = MaxPooling2D(pool_size=3, strides=2)(norm2)

    conv3 = Conv2D(filters=384, kernel_size=3, padding='same',
                   activation='relu')(pool2)

    conv4 = Conv2D(filters=384, kernel_size=3, padding='same',
                   activation='relu')(conv3)

    pad2 = ZeroPadding2D(padding=1)(conv4)
    conv5 = Conv2D(filters=256, kernel_size=2,
                   activation='relu')(pad2)

    pad3 = ZeroPadding2D(padding=1)(conv5)
    pool3 = MaxPooling2D(pool_size=3, strides=2)(pad3)

    pool4 = MaxPooling2D(pool_size=3, strides=2)(pool3)

    inception1 = inception(pool4, filters_batch=[64, 96, 128, 16, 32, 32])

    inception2 = inception(inception1, filters_batch=[
                           256, 160, 320, 32, 128, 128])

    pool5 = MaxPooling2D(pool_size=3, strides=2)(inception2)

    conv6 = Conv2D(filters=4096, kernel_size=1,
                   activation='relu')(pool5)

    flat = Flatten()(conv6)
    drop = Dropout(0.4)(flat)
    output = Dense(nClasses, activation='softmax',
                   kernel_regularizer=k_reg)(drop)

    model = Model(inputs=input, outputs=[output])
    return model


def modelInizialization():

    optimizer = optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    leave_model = createModel(input_shape)

    leave_model.compile(optimizer=optimizer,
                        loss="categorical_crossentropy", metrics=["accuracy"], loss_weights=[1.])

    leave_model.summary()

    return leave_model


def plotResult(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def modelTrain():

    model = modelInizialization()
    # Data Preparation
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = training_datagen.flow_from_directory(
        'dataset/training',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

    results = model.fit_generator(
        training_set,
        steps_per_epoch=1645,
        epochs=15,
        validation_data=test_set,
        validation_steps=412)

    plotResult(results)
    model.save("mango_model.h5")


def exit():
    n = 1


while n < 1:
    print("1. See Model Summary")
    print("2. Train Model")
    print("3. Exit")
    action = input("Choose an action:")

    if int(action) == 1:
        modelInizialization()
    elif int(action) == 2:
        modelTrain()
    else:
        n = 1
