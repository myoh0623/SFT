import tensorflow as tf
import bentoml
import PIL.Image as Image
import numpy as np

if __name__=="__main__":
    IMAGE_SHAPE = (224, 224)
    mobilenet_runner = bentoml.tensorflow.get("mobilenet:0.1").to_runner()
    mobilenet_runner.init_local()
    grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
    grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
    grace_hopper = np.array(grace_hopper)/255.0
    predict_data = grace_hopper[np.newaxis, ...].tolist()
    result = mobilenet_runner.run([predict_data])
    print(result)
    # Create a Runner instance:

    # iris_clf_runner = bentoml.tensorflow.get("iris_clf:latest").to_runner()

    # # Runner#init_local initializes the model in current process, this is meant for development and testing only:
    # iris_clf_runner.init_local()

    # # This should yield the same result as the loaded model:
    # print(iris_clf_runner.predict.run([[5.9, 3.0, 5.1, 1.8]]))