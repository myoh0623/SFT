import tensorflow as tf
import tensorflow_hub as hub

if __name__ == "__main__":
    hub_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    IMAGE_SHAPE = (224, 224)
    mobilenet_v2 = tf.keras.Sequential([
        hub.KerasLayer(hub_url, input_shape=IMAGE_SHAPE+(3,))
    ])
    model_path = './model'
    mobilenet_v2.save(model_path)
    tf.keras.utils.get_file('./model/ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    # new_model = tf.keras.models.load_model('saved_model/my_model')