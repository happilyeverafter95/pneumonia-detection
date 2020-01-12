import io
import tensorflow as tf

from fastapi import APIRouter, File
from PIL import Image
from keras.preprocessing.image import img_to_array

from classifier.train import Train

router = APIRouter()


@router.post('/predict')
def pnuemonia_router(image_file: bytes = File(...)):
    model = Train().define_model()
    model.load_weights('classifier/models/weights.h5')

    image = Image.open(io.BytesIO(image_file))

    if image.mode != 'L':
        image = image.convert('L')

    image = image.resize((64, 64))
    image = img_to_array(image)/255.0
    image = image.reshape(1, 64, 64, 1)

    graph = tf.get_default_graph()

    with graph.as_default():
        prediction = model.predict_proba(image)

    predicted_class = 'pneumonia' if prediction[0] > 0.5 else 'normal'

    return {'predicted_class': predicted_class,
            'pneumonia_probability': str(prediction[0])}
