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
    file_type = url.split('.')[-1]
    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    user_input_image = tf.keras.utils.get_file(f'{file_name}.{file_type}',url)
    user_input_image = Image.open(user_input_image).resize(IMAGE_SHAPE)
    image2array = np.array(user_input_image)/255.0
    return image2array[np.newaxis, ...].tolist()

def preprocessing_byte(byte_image):
    # 입력 받은 이미지 decoding
    decoded = base64.b64decode(byte_image)
    # BytesIO 를 이용해 image 를 open 한다. 
    IMAGE_SHAPE = (224, 224)
    user_input_image = Image.open(BytesIO(decoded)).resize(IMAGE_SHAPE)
    image2array = np.array(user_input_image)/255.0
    return image2array[np.newaxis, ...].tolist()


# 추론 결과 후처리 
def postprocessing(predicted_class):
    imagenet_labels = np.array(open('./ImageNetLabels.txt').read().splitlines())
    predicted_class_name = imagenet_labels[predicted_class]
    return predicted_class_name


def main():
    url = "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg"
    mobilenet_v2 = keras_model()
    image = preprocessing(url=url)
    result = mobilenet_v2.predict(image)
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class_name = postprocessing(predicted_class)
    print(predicted_class_name)
    return predicted_class_name

if __name__ =="__main__":
    main()