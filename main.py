import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt
import cv2

# Set page title
st.title("Disease Detection In Corn")
tab1, tab2 = st.tabs(["Predicted", "More"])

# Load model đã train trên colab
model = load_model('CNN_plant.h5')

# Danh sách các tên bệnh
class_name = ['Bệnh_đóm_lá_trên_ngô',
              'Bệnh_mốc_hạt_trên_ngô',
              'Bệnh_sỉ_sắt_trên_ngô',
              'Bệnh-sợi_đen_trên_ngô',
              'Bệnh_thiếu_đạm_trên_ngô',
              'Bệnh_thiếu_lân_trên_ngô',
              'Bệnh_ung_thư_trên_ngô']

# Function to preprocess image
def preprocess_image(image):
    # Đổi kích thước ảnh thành kích thước cố định
    image = image.resize((64, 64))
    # Chuyển đổi ảnh thành mảng numpy
    image_array = np.array(image)
    # Chuẩn hóa giá trị các pixel của ảnh
    image_array = image_array / 255.0
    # Mở rộng số chiều để phù hợp với kích thước đầu vào mong đợi của mô hình
    processed_image = np.expand_dims(image_array, axis=0)

    return processed_image

# Function to preprocess image and make predictions
def predict_object(image):
    # Dự đoán sử dụng mô hình
    prediction = model.predict(image)
    predicted_class = class_name[np.argmax(prediction)]
    confidence_scores = {class_name[i]: float(prediction[0][i]) for i in range(len(class_name))}
    return predicted_class, confidence_scores

# Function to open and read
def open_txt(file_path):
    # Mở file và đọc nội dung
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    # Hiển thị nội dung trong Streamlit
    st.text(content)

# Main application
def main():
    # Đọc ảnh đã tải lên
    image = Image.open(uploaded_file)

    # Hiển thị ảnh đã tải lên
    st.image(image, use_column_width=True)

    # Xử lý ảnh và dự đoán
    processed_image = preprocess_image(image)
    predicted_class, confidence_scores = predict_object(processed_image)

    # Hiển thị kết quả dự đoán
    st.success(f"Disease Detection: {predicted_class}")
    st.header("Chart of confidence")
    # danh sách độ tin cậy
    # for class_label, confidence in confidence_scores.items():
    #     st.write(f"- {class_label}: {confidence}")

    # Tạo danh sách nhãn lớp và độ tin cậy
    labels = list(confidence_scores.keys())
    confidences = list(confidence_scores.values())

    # Vẽ biểu đồ cột
    fig, ax = plt.subplots()
    ax.bar(labels, confidences)

    # Đặt tên cho trục x và y
    ax.set_xlabel('Disease in corn')
    ax.set_ylabel('Confidence')

    # Tăng khoảng cách giữa các nhãn trên trục x và xoay nhãn tránh chồng chéo
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)

# def main():
#     # Create a VideoCapture object to capture video from the camera
#     cap = cv2.VideoCapture(0)
#
#     # Set the frame size
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     # Run the application
#     stop_button_pressed = False  # Track if the stop button is pressed
#     while True:
#         # Read the current frame from the camera
#         ret, frame = cap.read()
#
#         # Convert the frame from BGR to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Convert the frame to PIL Image format
#         pil_image = Image.fromarray(frame_rgb)
#
#         # Preprocess the image and make predictions
#         processed_image = preprocess_image(pil_image)
#         predicted_class, confidence_scores = predict_object(processed_image)
#
#         # Display the current frame and predicted class label
#         st.image(frame_rgb, channels='RGB')
#         st.write("Predicted class:", predicted_class)
#
#         # Check for stop signal
#         if st.button('Stop', key='stop_button_unique'):
#             stop_button_pressed = True
#             break
#
#     # Release the VideoCapture object
#     cap.release()

with tab1:
    #Lựa chọn giữa 2 phương pháp
    selected = st.radio(
        "Choose method:",
        ('Image', 'Camera'))

    if selected == 'Image':
        # Tải lên ảnh
        uploaded_file = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'])
        if st.button('Predicted'):
            # Hiển thị thanh tiến trình
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)

            if uploaded_file is not None:
                # Chạy chương trình chính
                if __name__ == '__main__':
                    main()

    else:
        # Tải lên ảnh
        uploaded_file = st.camera_input("Take a picture")
        if st.button('Predicted'):
            # Hiển thị thanh tiến trình
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 1, text=progress_text)
            #Kiểm tra hình ảnh đã upload chưa
            if uploaded_file is not None:
                # Chạy chương trình chính
                if __name__ == '__main__':
                    main()

with tab2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        predicted_class, confidence_scores = predict_object(processed_image)

        if predicted_class == 'Bệnh_đóm_lá_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "domla.txt"
            open_txt(file_path)

        elif predicted_class == 'Bệnh_mốc_hạt_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "mochat.txt"
            open_txt(file_path)

        elif predicted_class == 'Bệnh_sỉ_sắt_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "sisat.txt"
            open_txt(file_path)

        elif predicted_class == 'Bệnh_sợi_đen_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "soiden.txt"
            open_txt(file_path)

        elif predicted_class == 'Bệnh_thiếu_đạm_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "thieudam.txt"
            open_txt(file_path)

        elif predicted_class == 'Bệnh_thiếu_lân_trên_ngô':
            # Đường dẫn tới file văn bản
            file_path = "thieulan.txt"
            open_txt(file_path)

        else :
            # Đường dẫn tới file văn bản
            file_path = "ungthu.txt"
            open_txt(file_path)

        # Search input
        search_query = st.text_input('Searching for more information about diseases:', predicted_class)

        # Perform search
        if st.button('Search on Youtube'):
            # Generate YouTube search URL
            youtube_url = f'https://www.youtube.com/results?search_query={search_query}'

            # Display link
            st.markdown(f'[YouTube Search Results]({youtube_url})')

        if st.button('Search on Google'):
            # Generate YouTube search URL
            google_search_url = f"https://www.google.com/search?q={search_query}"

            # Display link
            st.markdown(f'[Google Search Results]({google_search_url})')
