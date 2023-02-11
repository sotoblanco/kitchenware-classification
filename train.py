import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import os

df_extra = pd.read_csv('../input/kitchenware-extra-images/data.csv')
df_extra['filename'] = '../input/kitchenware-extra-images/data/' + df_extra['Id'] + '.jpg'

df_org = pd.read_csv('../input/kitchenware-classification/train.csv', dtype={'Id': str})
df_org['filename'] = '../input/kitchenware-classification/images/' + df_org['Id'] + '.jpg'

df = pd.concat([df_org, df_extra])

# Since we want to make sure to have a good validation schema 
# we split the data into training, validation and testing
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1,
    stratify=df['label']
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25,
    random_state=1, stratify=df_full_train['label']
)

def make_model(learning_rate, droprate, input_shape, inner_layer):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_shape, input_shape, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(input_shape, input_shape, 3))

    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    dense = keras.layers.Dense(inner_layer, activation='relu')(vectors)
    dropout = keras.layers.Dropout(droprate)(dense)
    outputs = keras.layers.Dense(6, activation="linear")(dropout)

    model = keras.Model(inputs, outputs)
    
    learning_rate = learning_rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    return model

config={
        "learning_rate":0.001,
        "droprate": 0.2,
        "input_shape":550,
        "inner_layer": 50,
        "epochs":100,
        "batch_size":32,
        "loss_function":"crossentropy",
        "architecture":"CNN",
        "dataset":"Kitchenware-plus-extra"
        }
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

train_full_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_full_generator = train_full_datagen.flow_from_dataframe(
    df_full_train,
    x_col='filename',
    y_col='label',
    target_size=(config.input_shape, config.input_shape),
    batch_size=config.batch_size,
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'kitchenware_v5_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'   
)

model = make_model(learning_rate=config.learning_rate,droprate=config.droprate,
                   input_shape=config.input_shape, inner_layer=config.inner_layer)

# add the wandbCallback
model.fit(
    train_full_generator,
    epochs=config.epochs
)

classes = np.array(list(train_full_generator.class_indices.keys()))

logits = model.predict(train_full_generator)
f_x = tf.nn.softmax(logits).numpy()

predictions = classes[f_x.argmax(axis=1)]