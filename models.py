import os
import numpy as np 

import tensorflow as tf 
from tf.keras.layers import Input, Dense, Conv2D, Dropout
from tf.keras.layers import Flatten, BatchNormalization
from tf.keras.layers import MaxPooling2D, AveragePooling2D
from tf.keras.layers import concatenate, Activation
from tf.keras.optimizers import Adam
from tf.keras.callbacks import ModelCheckpoint, EarlyStopping
from tf.keras.callbacks import ReduceLROnPlateau
from tf.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tf.keras.utils import plot_model, to_categorical
from tf.keras.models import Model


NUM_CLASSES = 225 
IMG_SHAPE = (112, 112, 3)
DENSE_BLOCKS_NUM = 3
GROWTH_RATE = 12
COMPRESSION_FACTOR = 0.5
DEPTH = 120
TARGET_SIZE = IMG_SHAPE[:2]
EPOCHS = 60
OUTPUT = './'
BATCH_SIZE = 16

def create_dense_block(filters, kernel_size, prev_layer, padding='same',
                      kernel_initializer='he_normal'):
    x = BatchNormalization()(prev_layer)
    x = Activation('relu')(x)
    return Conv2D(filters=filters, kernel_size=kernel_size, 
                  padding=padding, kernel_initializer=kernel_initializer)(x)
    


def create_densenet_bc(shape, num_classes, dense_blocks_num, 
                      depth, growth_rate, compression_factor):
    num_bottleneck_layers = (depth - 4) // (2 * dense_blocks_num)
    num_filters_before_dense_block = 2 * growth_rate
    
    inputs = Input(shape=shape)
    x = create_dense_block(num_filters_before_dense_block, 3, inputs)
    x = concatenate([inputs, x])
    
    for i in range(dense_blocks_num):
        for j in range(num_bottleneck_layers):
            y = create_dense_block(4*growth_rate, 1, x)
            y = Dropout(0.2)(y)
            y = create_dense_block(growth_rate, 3, y)
            y = Dropout(0.2)(y)
            x = concatenate([x, y])
            
        if i == dense_blocks_num - 1:
            continue
            
        num_filters_before_dense_block += num_bottleneck_layers * growth_rate
        num_filters_before_dense_block = int(num_filters_before_dense_block * compression_factor)
        
        y = BatchNormalization()(x)
        y = Conv2D(num_filters_before_dense_block, 1, padding='same',
                   kernel_initializer='he_normal')(y)
        x = AveragePooling2D()(y)
        
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, kernel_initializer='he_normal',
                    activation='softmax')(y)
    
    return Model(inputs=inputs, outputs=outputs)

image_model = create_densenet_bc(IMG_SHAPE, NUM_CLASSES, DENSE_BLOCKS_NUM, 
                           DEPTH, GROWTH_RATE, COMPRESSION_FACTOR)

image_model.load_weights("image_model.hdf5")