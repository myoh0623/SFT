import tensorflow as tf
import PIL.Image as Image
import numpy as np 
import datetime
import base64
from io import BytesIO

# load model
def keras_model():
    mobilenet_v2 = tf.keras.models.load_model('./model')
    return mobilenet_v2

# model 전처리
def preprocessing_url(url):
    IMAGE_SHAPE = (224, 224)
    user_input_image = tf.keras.utils.get_file(fname=url.split('/')[-1], origin=url)
    user_input_image = Image.open(user_input_image).resize(IMAGE_SHAPE)
    image2array = np.array(user_input_image) / 255.0
    return np.expand_dims(image2array, axis=0)  # 변경된 부분

def preprocessing_byte(byte_image):
    decoded = base64.b64decode(byte_image)
    IMAGE_SHAPE = (224, 224)
    user_input_image = Image.open(BytesIO(decoded)).resize(IMAGE_SHAPE)
    image2array = np.array(user_input_image) / 255.0
    return np.expand_dims(image2array, axis=0)  # 변경된 부분

# 추론 결과 후처리 
def postprocessing(predicted_class):
    imagenet_labels = np.array(open('./model/ImageNetLabels.txt').read().splitlines())
    predicted_class_name = imagenet_labels[predicted_class]
    return predicted_class_name

def main():
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    mobilenet_v2 = keras_model()
    image = preprocessing_url(url)  # 변경된 부분
    result = mobilenet_v2.predict(image)
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class_name = postprocessing(predicted_class)
    print(predicted_class_name)

if __name__ == "__main__":
    main()
