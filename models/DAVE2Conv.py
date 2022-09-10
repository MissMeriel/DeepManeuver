import argparse
import kornia
import numpy as np
import onnx
import skimage.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


from onnx2pytorch import ConvertModel
from pathlib import Path
from torchvision.transforms import Compose, ToPILImage, ToTensor, Resize
from torchvision.utils import save_image

import numpy as np
# import pandas as pd
# import cv2
import matplotlib.image as mpimg
import json

import h5py
import os
from PIL import Image
import PIL



class DAVE2ConvertedModel(nn.Module):

    def __init__(self, action_dim, max_action):
        def __init__(self):
            super().__init__()

            class LambdaLayer(nn.Module):
                def __init__(self, lambd=None):
                    super(self).__init__()
                    if lambd is not None:
                        self.lambd = lambd
                    else:
                        self.lambd = lambda x: x / 127.5 - 1.

                def forward(self, x):
                    return self.lambd(x)

            self.conv_layers = nn.Sequential(
                LambdaLayer(),
                nn.Conv2d(3, 24, 5, stride=2),
                nn.ELU(),
                nn.Conv2d(24, 36, 5, stride=2),
                nn.ELU(),
                nn.Conv2d(36, 48, 5, stride=2),
                nn.ELU(),
                nn.Conv2d(48, 64, 3),
                nn.ELU(),
                nn.Conv2d(64, 64, 3),
                nn.ELU()
                # TODO: Flatten layer here
            )
            self.dense_layers = nn.Sequential(
                # TODO: Add activation functions
                nn.Linear(in_features=1152, out_features=100),
                nn.Dropout(0.2),
                nn.ELU(),
                nn.Linear(in_features=100, out_features=50),
                nn.Dropout(0.2),
                nn.ELU(),
                nn.Linear(in_features=50, out_features=10),
                nn.Dropout(0.2),
                nn.ELU(),
                # modified to have 2 outputs
                nn.Linear(in_features=10, out_features=2)
            )

        def forward(self, data):
            # data = data.reshape(data.size(0), 1, 60, 120)
            # print(data.shape)
            output = self.conv_layers(data)
            # print(output.shape)
            output = output.view(output.size(0), -1)
            # print(output.shape)
            output = self.dense_layers(output)
            return output
        # TODO: check for atan output layer
        # self.model = Sequential()
        # self.input_shape = (150, 200, 3) #(960,1280,3)


    # outputs vector [steering, throttle]
    def define_dual_model_BeamNG(self):
        # Start of MODEL Definition
        self.model = Sequential()
        # Input normalization layer
        self.model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=self.input_shape, name='lambda_norm'))

        # 5x5 Convolutional layers with stride of 2x2
        self.model.add(Conv2D(24, 5, 2, name='conv1'))
        self.model.add(ELU(name='elu1'))
        self.model.add(Conv2D(36, 5, 2, name='conv2'))
        self.model.add(ELU(name='elu2'))
        self.model.add(Conv2D(48, 5, 2, name='conv3'))
        self.model.add(ELU(name='elu3'))

        # 3x3 Convolutional layers with stride of 1x1
        self.model.add(Conv2D(64, 3, 1, name='conv4'))
        self.model.add(ELU(name='elu4'))
        self.model.add(Conv2D(64, 3, 1, name='conv5'))
        self.model.add(ELU(name='elu5'))

        # Flatten before passing to the fully connected layers
        self.model.add(Flatten())
        # Three fully connected layers
        self.model.add(Dense(100, name='fc1'))
        self.model.add(Dropout(.5, name='do1'))
        self.model.add(ELU(name='elu6'))
        self.model.add(Dense(50, name='fc2'))
        self.model.add(Dropout(.5, name='do2'))
        self.model.add(ELU(name='elu7'))
        self.model.add(Dense(10, name='fc3'))
        self.model.add(Dropout(.5, name='do3'))
        self.model.add(ELU(name='elu8'))

        # Output layer with tanh activation
        self.model.add(Dense(2, activation='tanh', name='output'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer="adam", loss="mse")
        # self.load_weights(h5_filename)
        return self.model

    def atan_layer(self, x):
        return tf.multiply(tf.atan(x), 2)

    def atan_layer_shape(self, input_shape):
        return input_shape

    def define_model_DAVEorig(self):
        input_tensor = Input(shape=(100, 100, 3))
        model = Sequential()
        model.add(Conv2D(24, 5, 2, padding='valid', activation='relu', name='block1_conv1'))
        model.add(Conv2D(36, 5, 2, padding='valid', activation='relu',  name='block1_conv2'))
        model.add(Conv2D(48, 5, 2, padding='valid', activation='relu', name='block1_conv3'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv4'))
        model.add(Conv2D(64, 3, 1, padding='valid', activation='relu', name='block1_conv5'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(1164, activation='relu', name='fc1'))
        model.add(Dense(100, activation='relu', name='fc2'))
        model.add(Dense(50, activation='relu', name='fc3'))
        model.add(Dense(10, activation='relu', name='fc4'))
        model.add(Dense(1, name='before_prediction'))
        model.add(Lambda(lambda x: tf.multiply(tf.atan(x), 2), output_shape=input_tensor, name='prediction'))
        model.compile(loss='mse', optimizer='adadelta')
        self.model = model
        return model

    def load_weights(self, h5_file):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = '{}/{}'.format(dir_path, h5_file)
        self.model.load_weights(filename)
        return self.model

    def process_image(self, image):
        # image = image.crop((0, 200, 512, 369))
        # image = image.resize((self.input_shape[1], self.input_shape[0]), Image.ANTIALIAS)
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image = np.array(image).reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return image

    # Functions to read and preprocess images
    def readProcess(self, image_file):
        """Function to read an image file and crop and resize it for input layer

        Args:
          image_file (str): Image filename (expected in 'data/' subdirectory)

        Returns:
          numpy array of size 66x200x3, for the image that was read from disk
        """
        # Read file from disk
        image = mpimg.imread('data/' + image_file.strip())
        # Remove the top 20 and bottom 20 pixels of 160x320x3 images
        image = image[20:140, :, :]
        # Resize the image to match input layer of the model
        resize = (self.input_shape[0], self.input_shape[1])
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
        return image

    def randBright(self, image, br=0.25):
        """Function to randomly change the brightness of an image

        Args:
          image (numpy array): RGB array of input image
          br (float): V-channel will be scaled by a random between br to 1+br

        Returns:
          numpy array of brighness adjusted RGB image of same size as input
        """
        rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rand_bright = br + np.random.uniform()
        rand_image[:,:,2] = rand_image[:,:,2]*rand_bright
        rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
        return rand_image
