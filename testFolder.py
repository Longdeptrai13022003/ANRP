import re
import tkinter as tk
from tkinter import filedialog, Label, Button
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import easyocr
import os
import time

# Khởi tạo YOLO model và EasyOCR
model = YOLO("runs/detect/train5/weights/best.pt")  # Đường dẫn tới best.pt
reader = easyocr.Reader(['en'], gpu=False)  # EasyOCR, dùng CPU để nhận diện ký tự

# Tạo giao diện Tkinter
window = tk.Tk()
window.title("License Plate Recognition")
window.geometry("1080x720")  # Kích thước cửa sổ
window.config(bg="#f5f5f5")  # Màu nền sáng

title_label = Label(window, text="Automatic Vietnamese car number license recognition", bg="#f5f5f5",
                    font=("Arial", 20, "bold"))
title_label.pack(pady=10)

# Khởi tạo biến ảnh để hiển thị (cần phải khởi tạo sau khi cửa sổ chính đã được tạo)
img_label = Label(window, bg="#f5f5f5")
img_label.pack(pady=20)

footer_label = Label(window, text="Student: Tran Duc Long | MSV: 90544", bg="#f5f5f5", font=("Arial", 12))
footer_label.pack(side="bottom", pady=10)

# Tạo thư mục output nếu chưa có
if not os.path.exists("output"):
    os.makedirs("output")

# Biến kiểm tra việc video đang chạy hay không
is_running = True


# Hàm cập nhật ảnh trên giao diện Tkinter
def update_image(frame, img_label):
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    img_label.config(width=frame.shape[1], height=frame.shape[0])
    img_label.configure(image=img)
    img_label.image = img


# Hàm xử lý nhận diện trên ảnh hoặc video
def process_image_video(path):
    global is_running  # Sử dụng biến toàn cục
    cap = cv2.VideoCapture(path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window.geometry(f"{width}x{height}")  # Cập nhật kích thước cửa sổ
    frame_count = 0
    fps = 30
    frame_delay = 1 / fps
    playback_speed = 1
    adjusted_frame_delay = frame_delay * playback_speed
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join("output", timestamp)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)
        detected_plate = False

        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
            plate_img = frame[y_min:y_max, x_min:x_max]
            ocr_result = reader.readtext(plate_img)
            if ocr_result:
                plate_text1 = ' '.join([text[1] for text in ocr_result])
                plate_text = re.sub(r'[^0-9a-zA-Z-]', '', plate_text1)
                confidence = sum([text[2] for text in ocr_result]) / len(ocr_result) * 100

                label = f"{plate_text} ({confidence:.2f}%)"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                detected_plate = True

        if detected_plate:
            output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_filename, frame)

        update_image(frame, img_label)
        window.update_idletasks()
        window.update()
        time.sleep(adjusted_frame_delay)

    cap.release()


# Hàm chọn file ảnh hoặc video từ máy tính
def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("Video files", "*.mp4 *.avi")])
    if file_path:
        process_image_video(file_path)


# Hàm sử dụng camera laptop
def use_camera():
    global is_running
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window.geometry(f"{width}x{height}")
    frame_count = 0
    fps = 30
    frame_delay = 1 / fps
    playback_speed = 0.5
    adjusted_frame_delay = frame_delay * playback_speed
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join("output", timestamp)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame)
        detected_plate = False
        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
            plate_img = frame[y_min:y_max, x_min:x_max]
            ocr_result = reader.readtext(plate_img)
            if ocr_result:
                plate_text1 = ' '.join([text[1] for text in ocr_result])
                plate_text = re.sub(r'[^0-9a-zA-Z-]', '', plate_text1)
                confidence = sum([text[2] for text in ocr_result]) / len(ocr_result) * 100
                label = f"{plate_text} ({confidence:.2f}%)"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                detected_plate = True
        if detected_plate:
            output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_filename, frame)
        update_image(frame, img_label)
        window.update_idletasks()
        window.update()
        time.sleep(adjusted_frame_delay)

    cap.release()


# Hàm xử lý phím tắt
def handle_key(event):
    global is_running
    if event.char == 't':  # Phím 'T' để chọn file mới
        select_file()
    elif event.char == 'q':  # Phím 'Q' để dừng video
        is_running = False  # Dừng vòng lặp video
        window.quit()  # Thoát khỏi cửa sổ


# Gán phím tắt
window.bind('<KeyPress>', handle_key)


# Hàm chọn thư mục chứa ảnh
def select_directory():
    folder_path = filedialog.askdirectory()  # Chọn thư mục
    if folder_path:
        process_images_in_directory(folder_path)  # Gọi hàm xử lý ảnh trong thư mục đã chọn


# Hàm xử lý tất cả ảnh trong thư mục
def process_images_in_directory(directory_path):
    global is_running  # Sử dụng biến toàn cục

    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Kiểm tra nếu tệp là ảnh
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4')):
            print(f"Processing image: {filename}")
            process_image_video(file_path)  # Gọi hàm xử lý ảnh/video đã có


# Tạo các nút chọn ảnh/video và camera
select_button = Button(window, text="Select Image/Video", command=select_file, bg="#008000", font=("Arial", 18))
select_button.pack(pady=20)

camera_button = Button(window, text="Use Camera", command=use_camera, bg="#008000", font=("Arial", 18))
camera_button.pack(pady=20)

# Tạo nút chọn thư mục
select_directory_button = Button(window, text="Select Folder of Images", command=select_directory, bg="#008000",
                                 font=("Arial", 18))
select_directory_button.pack(pady=20)

window.mainloop()
