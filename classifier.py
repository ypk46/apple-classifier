import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

nClasses = 2
input_shape = (256, 256, 3)


def createModel():
    model = Sequential()

    model.add(Conv2D(filters=96, kernel_size=9, strides=4,
                     activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=256, kernel_size=5, strides=1, activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"))
    model.add(Conv2D(filters=384, kernel_size=3, strides=1, activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=2, strides=1, activation="relu"))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Conv2D(filters=4096, kernel_size=1, strides=1, activation="relu"))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(nClasses, activation="softmax"))

    return model


# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical')

leave_model = createModel()
leave_model.compile(optimizer="rmsprop",
                    loss="categorical_crossentropy", metrics=["accuracy"])
leave_model.summary()

history = leave_model.fit_generator(
    training_set,
    steps_per_epoch=200,
    epochs=20,
    validation_data=test_set,
    validation_steps=20)

leave_model.save("mango_model.h5")
