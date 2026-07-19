
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tempfile
import time

# 모델 로드
@st.cache_resource
def load_cnn_model():
    return load_model("Cats_and_Dogs.keras")

model = load_cnn_model()

# 예측 함수
def predict(img_array):
    img_resized = cv2.resize(img_array, (150, 150))  # 모델 입력 사이즈 맞추기
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return "강아지 🐶" if pred[0][0] > 0.5 else "고양이 🐱"

# Streamlit UI
st.title("고양이 vs. 강아지 분류기 🐱🐶")
st.write("이미지 또는 비디오를 업로드하거나 웹캠을 통해 촬영하세요.")

option = st.radio("입력 방식 선택", ['이미지 업로드', '웹캠 촬영', '영상 업로드'])

if option == '이미지 업로드':
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="업로드된 이미지", use_container_width=True)
        img_array = np.array(img)
        result = predict(img_array)
        st.success(f"예측 결과: **{result}**")

elif option == '웹캠 촬영':
    cap = st.camera_input("웹캠으로 이미지 촬영")
    if cap:
        img = Image.open(cap)
        st.image(img, caption="촬영된 이미지", use_container_width=True)
        img_array = np.array(img)
        result = predict(img_array)
        st.success(f"예측 결과: **{result}**")
        # 실시간 웹캠 스트리밍은 Streamlit 기본 기능으로는 지원되지 않음

elif option == '영상 업로드':
    video_file = st.file_uploader("비디오 파일을 업로드하세요", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            result = predict(frame)
            frame_placeholder.image(frame, caption=f"Frame {frame_count}", use_container_width=True)
            result_placeholder.success(f"예측 결과: **{result}**")
            # 프레임 간 간격을 조정 (너무 빠르면 UI가 갱신되지 않으므로 약간의 sleep)
            time.sleep(0.05)
        cap.release()
        st.info(f"총 {frame_count} 프레임을 처리했습니다.")
