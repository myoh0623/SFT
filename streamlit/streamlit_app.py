import streamlit as st
import requests
import base64 #파이너리 파일을 전송하기 위해 encoding 해준다. 서버에서 decoding 해준다. 
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--add", dest="address", action="store") 
# args = parser.parse_args()
# address = args.address
address = "flask_app"

if __name__ == "__main__":
    try:
        image = st.file_uploader("upload image", type=["jpg", "png", "jpeg"])
        if image:
            # 화면에 출력
            st.image(image)
            # 업로드한 이미지를 base64로 인코딩
            img_bytes = image.read()
            encoded = base64.b64encode(img_bytes)
            # print(encoded)
            url = f'http://{address}:5000/predict' #localhost = 127.0.0.1
            data = {"image_data" : encoded}
            response = requests.post(url, data = data)
            print(response)
            st.write("분석 결과 : ", response.json()["class"])
    except:
        st.write("실패하였습니다.")