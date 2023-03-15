import tensorflow as tf
import tensorflow_hub as hub

import bentoml


if __name__=="__main__":
    # Load training data
    hub_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    IMAGE_SHAPE = (224, 224)
    mobilenet_v2 = tf.keras.Sequential([
        hub.KerasLayer(hub_url, input_shape=IMAGE_SHAPE+(3,))
    ])
    bentoml.tensorflow.save_model(
        mobilenet_v2,
        "mobilenet:0.1",
        signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )