import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data
import os
from tqdm import *
import struct
from collections import defaultdict
import cv2
from keras import backend as K
import threading
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from skimage.color import rgb2yuv

# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from sklearn.metrics import log_loss

#from load_cifar10 import load_cifar10_data

import sys
sys.setrecursionlimit(3000)

############################################################################
__GLOBAL_PARAMS__ = {
    'MODEL' : "resnet101" ,
    'DEBUG' : True,
    'NORMALIZATION' : True,
    'YUV' : True ,
    'MULTI_SCALE' : False
}
########

if __GLOBAL_PARAMS__['MULTI_SCALE']:
    raise Exception("MULTI_SCALE not supported yet!")

__MODEL__KEY__ = ""
for k in sorted(__GLOBAL_PARAMS__.keys()):
    if not k.startswith("_"):
        __MODEL__KEY__ += "__" + str(k) + "_" + str(__GLOBAL_PARAMS__[k])

if (__GLOBAL_PARAMS__['DEBUG']):
    LOG_FILE = "simple.log"
else:
    LOG_FILE = "log" + __MODEL__KEY__ + ".log"

SUB_FILE = "sub" + __MODEL__KEY__ + ".csv.gz"

import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename=LOG_FILE,level=logging.DEBUG)
#logging.debug('This message should go to the log file')
if (__GLOBAL_PARAMS__['DEBUG']):
    logging.info('** DEBUG: '+__MODEL__KEY__+' ****************************************************************')
else:
    logging.info('** PRODUCTION:'+__MODEL__KEY__+' ****************************************************************')

#logging.warning('And this, too')

########### -------------> FUNC

from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        #self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        #self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet101_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
      print(">>>>>>>>>>>>>>>>>. Custom Input <<<<<<<<<<<<<<<")
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,4):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,23):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((6, 6), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'resnet101_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'resnet101_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    #x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_image(x):
    if __GLOBAL_PARAMS__['NORMALIZATION']:
        x = (x - 128.0) / 128.0
    if __GLOBAL_PARAMS__['YUV']:
        x = np.array([rgb2yuv(x.reshape((1,180,180,3)))])
        x = x.reshape((180,180,3))
    return x

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180),
                 with_labels=True, batch_size=32, shuffle=False, seed=None):
        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))
        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]
                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])
            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]
            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            # Preprocess the image.
            x = img_to_array(img)
            x = preprocess_image(x)
            #x = self.image_data_generator.random_transform(x)
            #x = self.image_data_generator.standardize(x)
            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1
        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x
    def next(self):
        with self.lock:
            index_array  = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array[0])

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break
            length = struct.unpack("<i", item_length_bytes)[0]
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length
            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])
            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row
            offset += length
            f.seek(offset)
            pbar.update()
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])
    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]
            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replace=False)
            # Randomly choose the products that become part of the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []
            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


########### -------------> MAIN
categories_path = os.path.join("data", "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
print(">>> Mapping category_id to an integer index ... ")
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)
print(categories_df.head())

cat2idx, idx2cat = make_category_tables()
# Test if it works:
print(cat2idx[1000012755], idx2cat[4] , len(cat2idx))

print(">>> Train set ... ")
data_dir = "data"

if (__GLOBAL_PARAMS__['DEBUG']):
    print(">>> DEBUG mode ... ")
    train_bson_path = os.path.join(data_dir, "train_example.bson")
    num_train_products = 82
else:
    print(">>> PRODUCTION mode ... ")
    train_bson_path = os.path.join(data_dir, "train.bson")
    num_train_products = 7069896
test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182
print(train_bson_path,num_train_products)


if (not __GLOBAL_PARAMS__['DEBUG']):
    if os.path.isfile("train_offsets.csv"):
        print(">> reading from file train_offsets ... ")
        train_offsets_df = pd.read_csv("train_offsets.csv")
        train_offsets_df.set_index( "product_id" , inplace= True)
        train_offsets_df.sort_index(inplace=True)
    else:
        train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
        train_offsets_df.to_csv("train_offsets.csv")
    print(train_offsets_df.head())

    if os.path.isfile("train_images.csv"):
        print(">> reading from file train_images / val_images ... ")
        train_images_df = pd.read_csv("train_images.csv")
        train_images_df = train_images_df[['product_id','category_idx','img_idx']]
        val_images_df = pd.read_csv("val_images.csv")
        val_images_df = val_images_df[['product_id', 'category_idx', 'img_idx']]
    else:
        train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, drop_percentage=0)
        train_images_df.to_csv("train_images.csv")
        val_images_df.to_csv("val_images.csv")
    print(train_images_df.head())
    print(val_images_df.head())
    categories_df.to_csv("categories.csv")
else:
    train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
    train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, drop_percentage=0)
    print(train_images_df.head())
    print(val_images_df.head()) 

## Generator
print(">>> Generator ... ")
# Tip: use ImageDataGenerator for data augmentation and preprocessing ??
train_bson_file = open(train_bson_path, "rb")
lock = threading.Lock()
num_classes = len(cat2idx)
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 16

train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)


## Model
img_rows, img_cols = 180, 180 # Resolution of inputs
channel = 3
nb_epoch = 10

# Load Cifar10 data. Please implement your own load_data() module for your own dataset
#X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

# Load our model
model = resnet101_model(img_rows, img_cols, channel, num_classes)
model.summary()

# Start Fine-tuning
early_stopping = EarlyStopping(monitor='val_loss', patience=0 )
bst_model_path = "mod" + __MODEL__KEY__ + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

# To train the model:
history  = model.fit_generator(train_gen,
                    epochs=200,
                    steps_per_epoch = num_train_images // batch_size + 1,
                    validation_data = val_gen,
                    validation_steps = num_val_images // batch_size + 1,
                    callbacks=[early_stopping, model_checkpoint])

print(history.history.keys())

# logging
logging.info('N. epochs == '+str(len(history.history['val_acc'])))
logging.info('Val accuracy == '+str(max(history.history['val_acc'])))

## Predict on Test-set
print(">>> Predicting on test-set ... ")
submission_df = pd.read_csv("data/sample_submission.csv")
print(submission_df.head())
test_datagen = ImageDataGenerator()
data = bson.decode_file_iter(open(test_bson_path, "rb"))
with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])
        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())
        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]
            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = preprocess_image(x)
            # = test_datagen.random_transform(x)
            # = test_datagen.standardize(x)
            # Add the image to the batch.
            batch_x[i] = x
        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)
        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]
        pbar.update()
submission_df.to_csv(SUB_FILE, compression="gzip", index=False)
