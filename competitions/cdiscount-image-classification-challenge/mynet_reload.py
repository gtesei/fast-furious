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

############################################################################
__GLOBAL_PARAMS__ = {
    'MODEL' : "mynet" ,
    'DEBUG' : False,
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

SUB_FILE = "sub_reloaded_" + __MODEL__KEY__ + ".csv.gz"

import logging
logging.basicConfig(format='%(asctime)s %(message)s', filename=LOG_FILE,level=logging.DEBUG)
#logging.debug('This message should go to the log file')
if (__GLOBAL_PARAMS__['DEBUG']):
    logging.info('** DEBUG: '+__MODEL__KEY__+' ****************************************************************')
else:
    logging.info('** PRODUCTION:'+__MODEL__KEY__+' ****************************************************************')

#logging.warning('And this, too')

########### -------------> FUNC
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
batch_size = 128

train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)


## Model
print(">>> reloading last model ... ")
print("mod" + __MODEL__KEY__ + '.h5')
from keras.models import load_model

inputs = Input(shape=(180, 180, 3))
x = Conv2D(32, 5, padding="valid", activation="relu")(inputs)
x = BatchNormalization()(x) # hope similar to local response normalization
x = MaxPooling2D(pool_size=(3, 3))(x)
#fl1 = Flatten()(x)

x2 = Conv2D(64, 5, padding="valid", activation="relu")(x)
x2 = BatchNormalization()(x2) # hope similar to local response normalization
x2 = MaxPooling2D(pool_size=(3, 3))(x2)
fl2 = Flatten()(x2)

#merged = concatenate([fl1, fl2]) # multi scale features
merged = Dropout(0.5)(fl2)
merged = BatchNormalization()(merged)

merged = Dense(2*num_classes, activation='relu')(merged)
merged = Dropout(0.5)(merged)
merged = BatchNormalization()(merged)
predictions = Dense(num_classes, activation='softmax')(merged)

model = Model(inputs=inputs, outputs=predictions)


model.load_weights("mod" + __MODEL__KEY__ + '.h5')

model.summary()

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
