# import the libraries
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('kitchenware_v4_09_0.967.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('kitchenware-class.tflite', 'wb') as f_out:
    f_out.write(tflite_model)