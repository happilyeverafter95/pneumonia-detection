import io
import numpy as np
import tensorflow as tf

from fastapi import APIRouter, File
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from classifier.train import Train

router = APIRouter()


@router.post('/predict')
def skin_lesion_classification(image_file: bytes = File(...)):
    model = Train().define_model()
    model.load_weights('classifier/data/models/weights.h5')

    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'LA':
        image = image.convert('LA')

    image = image.resize((64, 64))

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    graph = tf.get_default_graph()

    with graph.as_default():
        predicted_class = model.predict_proba(image)

    predicted_class = 'pneumonia' if predicted_class[0] > 0.5 else 'normal'

    return {'predicted_class': predicted_class, 
            'pneumonia_probability': str(predicted_class[0])}
