# from flask import Flask, request
from flask import Flask, request, jsonify
from load_model import keras_model, preprocessing_byte, postprocessing
import json
import numpy as np
import base64

app = Flask(__name__)

@app.route("/predict", methods=["POST"]) 
def inference():
    # # url 을 입력 받았을때 출력하는 code
    # mobilenet_v2 = keras_model()
    # request_body = request.get_json()
    # user_url =  request_body["url"]
    # image = preprocessing(url=user_url)
    # result = mobilenet_v2.predict(image)
    # #################################
    mobilenet_v2 = keras_model()
    image_data = request.form.get("image_data")
    image = preprocessing_byte(image_data)
    result = mobilenet_v2.predict(image)
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class_name = postprocessing(predicted_class)
    result = {"predicted":str(result[0]), "class":predicted_class_name}
    print(predicted_class_name)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)