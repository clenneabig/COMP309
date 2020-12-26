#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import tensorflow as tf
import random
import pickle

# Set random seeds to ensure the reproducible results


SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)
tf.random.set_seed(SEED)


def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    """model = Sequential()

    model.add(
        Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", input_shape=(300, 300, 3),
               data_format="channels_last", activation='sigmoid'))

    model.add(MaxPooling2D(pool_size=2, strides=1, padding="valid", data_format="channels_last"))

    model.add(
        Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", data_format="channels_last",
         activation='sigmoid'))

    model.add(Dropout(0.1))

    model.add(
        Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", data_format="channels_last", activation='sigmoid'))

    model.add(MaxPooling2D(pool_size=2, strides=1, padding="valid", data_format="channels_last"))

    model.add(Flatten())

    model.add(Dense(units=64, activation='relu'))

    model.add(Dense(units=3, activation='softmax'))"""

    model = VGG16(include_top=False, input_shape=(300, 300, 3), weights='imagenet')

    for layer in model.layers:
        layer.trainable = False

    drop = Dropout(0.1)(model.output)
    conv = Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", data_format="channels_last",
                  activation='relu')(drop)
    pool = MaxPooling2D(pool_size=2, strides=1, padding="valid", data_format="channels_last")(conv)
    flat = Flatten()(pool)
    dense = Dense(units=64, activation='relu')(flat)
    output = Dense(units=3, activation='softmax')(dense)

    model = Model(inputs=model.inputs, outputs=output)

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


    return model


def train_model(model, flow, val):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    # Add your code here
    mc = ModelCheckpoint('model/gggggggmodel.h5', monitor='val_loss', save_best_only=True)
    hist = model.fit_generator(flow, epochs=100, verbose=1, callbacks=[mc], validation_data=val)
    with open('ggggggggggggggGraphData.pkl', 'wb') as f:
        pickle.dump(hist.history, f)
    return model


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = list(paths.list_images(test_data_dir))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    images_data = np.asarray(images_data)

    rows = images_data.shape[1]
    cols = images_data.shape[2]
    channels = images_data.shape[3]

    if K.image_data_format() == 'channels_first':
        images_data = images_data.reshape(images_data.shape[0], channels, rows, cols)
    else:
        images_data = images_data.reshape(images_data.shape[0], rows, cols, channels)

    images_data = images_data.astype('float32')
    images_data /= 255

    lb = LabelBinarizer()
    y = lb.fit_transform(labels)

    train_x, test_x, train_y, test_y = train_test_split(images_data, y, test_size=0.3, random_state=309)

    datagen = ImageDataGenerator(width_shift_range=150, height_shift_range=150, horizontal_flip=True)
    flow = datagen.flow(train_x, train_y, batch_size=64, seed=309)
    return flow, (test_x, test_y)


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    #model.save("model/model.h5")
    print("Model Saved Successfully.")


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/Train_data",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args


if __name__ == '__main__':
    model = construct_model()
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (300, 300)

    # Load images
    flow, (x_val, y_val) = load_images(test_data_dir, image_size)

    model = train_model(model, flow, (x_val, y_val))
    save_model(model)
