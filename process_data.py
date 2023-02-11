import tflite_runtime.interpreter as tflite

import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'kitchenware-class.tflite')

classes = np.array(['cup', 'fork', 'glass', 'knife', 'plate', 'spoon'])

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(550, 550))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    predictions = classes[preds.argmax(axis=1)]

    return predictions[0]


#def lambda_handler(event, context):
#    url = event['url']
#    pred = predict(url)
#    result = np.where(pred > 0.8, 'Pneumonia', 'Normal')

 #   return {'prediction':result}


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)

    return pred

event = {'url':"data:image/jpeg;base64,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" }

print(lambda_handler(event, None))