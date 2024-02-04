import streamlit as st
import requests
import base64

if __name__ == "__main__":
    try:
        image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])
        if image:
            st.image(image)
            img_bytes = image.read()
            encoded = base64.b64encode(img_bytes).decode('utf-8')  # bytes를 문자열로 변환
            url = 'http://172.20.15.246:5000/predict'
            data = {"image_data": encoded}
            response = requests.post(url, data=data)
            if response.ok:
                st.write("분석 결과 : ", response.json()["class"])
            else:
                st.write("서버 오류로 인해 분석을 완료할 수 없습니다.")
    except Exception as e:
        st.write("오류가 발생하였습니다:", str(e))
