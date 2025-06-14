import re
import tkinter as tk
from tkinter import filedialog, Label, Button
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import easyocr
import numpy as np
import os

# Khởi tạo YOLO model và EasyOCR
model = YOLO("runs/detect/train5/weights/best.pt")  # Đường dẫn tới best.pt
reader = easyocr.Reader(['en'], gpu=False)  # EasyOCR, dùng CPU để nhận diện ký tự
plate_pattern = re.compile(r'^\d{2}[A-Z]([-|\s|\d])[\d\.]+$')

# Tạo thư mục lưu ảnh output
os.makedirs("output_images", exist_ok=True)

# Tạo giao diện Tkinter
window = tk.Tk()
window.title("License Plate Recognition")
window.geometry("1080x720")
window.config(bg="#f5f5f5")

# Tạo các thành phần giao diện
title_label = Label(window, text="Automatic Vietnamese car number license recognition", bg="#f5f5f5",
                    font=("Helvetica", 16))
title_label.pack()

image_label = Label(window)
image_label.pack()


# Hàm để hiển thị ảnh lên giao diện Tkinter
def display_image(image, title):
    img = Image.fromarray(image)
    img.thumbnail((640, 480))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    image_label.config(text=title)


# Hàm chọn ảnh từ máy tính
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)


# Hàm xử lý ảnh đầu vào và nhận diện biển số xe
def process_image(file_path):
    # Đọc ảnh và hiển thị ảnh gốc
    image = cv2.imread(file_path)
    display_image(image, "Original Image")

    # Chuyển đổi kênh màu RGB -> BGR (theo yêu cầu của YOLO) và lưu lại
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output_images/color_converted_for_yolo.jpg", bgr_image)
    display_image(bgr_image, "Color Converted for YOLO")

    # Phát hiện đối tượng bằng YOLO
    results = model(bgr_image)
    plates = []
    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate_img = bgr_image[y1:y2, x1:x2]
            plates.append(plate_img)
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Lưu ảnh có khung biển số sau khi qua YOLO và hiển thị
    cv2.imwrite("output_images/detected_plate_after_yolo.jpg", bgr_image)
    display_image(bgr_image, "Detected License Plate after YOLO")

    # Tiền xử lý và nhận diện ký tự trên từng biển số
    for i, plate_img in enumerate(plates):
        # Tiền xử lý với lọc nhiễu và tách ngưỡng
        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        denoised_plate = cv2.fastNlMeansDenoising(gray_plate, None, 30, 7, 21)
        _, threshold_plate = cv2.threshold(denoised_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Lưu và hiển thị ảnh sau khi lọc nhiễu và tách ngưỡng
        cv2.imwrite(f"output_images/plate_denoised_threshold_{i + 1}.jpg", threshold_plate)
        display_image(threshold_plate, f"Preprocessed Plate {i + 1}")

        # Nhận diện ký tự với EasyOCR
        result_text = reader.readtext(threshold_plate)
        recognized_text = ""
        for bbox, text, prob in result_text:
            if plate_pattern.match(text):
                recognized_text = text
                break

        # In chuỗi ký tự nhận diện được ra console
        print(f"Recognized text for plate {i + 1}: {recognized_text}")

        # Hiển thị ảnh biển số với chữ nhận diện được và lưu lại
        cv2.putText(plate_img, recognized_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(f"output_images/recognized_plate_{i + 1}.jpg", plate_img)
        display_image(plate_img, f"Recognized Plate {i + 1} - Text: {recognized_text}")


# Nút chọn ảnh
select_image_btn = Button(window, text="Select Image", command=select_image, bg="#4CAF50", fg="white",
                          font=("Helvetica", 12))
select_image_btn.pack(pady=20)

window.mainloop()
