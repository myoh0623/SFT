import streamlit as st
import requests
import base64 #파이너리 파일을 전송하기 위해 encoding 해준다. 서버에서 decoding 해준다. 
from io import BytesIO
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--add", dest="address", action="store") 
# args = parser.parse_args()
# address = args.address
import numpy as np 
from PIL import Image


def preprocessing_byte(byte_image):
    # 입력 받은 이미지 decoding
    # BytesIO 를 이용해 image 를 open 한다. 
    IMAGE_SHAPE = (224, 224)
    user_input_image = Image.open(byte_image).resize(IMAGE_SHAPE)
    image2array = np.array(user_input_image)/255.0
    return image2array[np.newaxis, ...].tolist()


if __name__ == "__main__":
    try:
        image = st.file_uploader("upload image", type=["jpg", "png", "jpeg"])
        if image:
            # 화면에 출력
            st.image(image)
            # 업로드한 이미지를 base64로 인코딩
            pillow_img = Image.open(image)
            IMAGE_SHAPE = (224, 224)
            image = np.array(pillow_img.resize(IMAGE_SHAPE)).astype("float32")
            input_arr = np.array(image)/255.0
            predict_data = str(input_arr[np.newaxis, ...].tolist()) # post 로 전송하기 위해서 문자열로 변경 

            # data = '[1,2,3,4]' # 문자열로 변경

            content_type = "application/json"
            accept = "application/json"
            address = "127.0.0.1"
            url = f'http://{address}:5002/predict' #localhost = 127.0.0.1
            response = requests.post(url, data = predict_data, headers={"content-type" : content_type, "accept": accept})
            print(response.json(), type(response.json()))
            st.write("분석 결과 : ", response.json()["class"])
    except Exception as e:
        st.write(e)