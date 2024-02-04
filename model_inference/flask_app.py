from flask import Flask, request, jsonify
from load_model import keras_model, preprocessing_byte, postprocessing
import numpy as np

app = Flask(__name__)
mobilenet_v2 = keras_model()  # 모델을 전역 변수로 한 번만 로드

@app.route("/predict", methods=["POST"])
def inference():
    image_data = request.form.get("image_data")
    image = preprocessing_byte(image_data)
    result = mobilenet_v2.predict(image)
    predicted_class = np.argmax(result[0], axis=-1)
    predicted_class_name = postprocessing(predicted_class)
    result = {"predicted": str(result[0]), "class": predicted_class_name}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
