import re
import tkinter as tk
from tkinter import filedialog, Label, Button
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import easyocr
import os
import time
from sort.sort import Sort
import numpy as np
tracker = Sort()

model = YOLO("runs/detect/train8/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)
# Biểu thức chính quy cho định dạng biển số
plate_pattern = re.compile(r'^\d{2}[A-Z]([-|\s|\d])[\d\.]+$')  # 2 chữ số, 1 chữ cái, dấu '-' hoặc dấu cách hoặc chữ số, và các chữ số

window = tk.Tk()
window.title("License Plate Recognition")
window.geometry("1080x720")
window.config(bg="#f5f5f5")



title_label = Label(window, text="Automatic Vietnamese car license plate recognition", bg="#f5f5f5",
                    font=("Arial", 20, "bold"))
title_label.pack(pady=10)

img_label = Label(window, bg="#f5f5f5")
img_label.pack(pady=20)

footer_label = Label(window, text="Student: Tran Duc Long | MSV: 90544", bg="#f5f5f5", font=("Arial", 12))
footer_label.pack(side="bottom", pady=10)

# Tạo thư mục output nếu chưa có
if not os.path.exists("output"):
    os.makedirs("output")


is_running = True



def update_image(frame, img_label):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img = ImageTk.PhotoImage(img)
    img_label.config(width=frame.shape[1], height=frame.shape[0])
    img_label.configure(image=img)
    img_label.image = img

def refine_ocr_text(text):
    if len(text) >= 3:

        if text[2] == '4':
            text = text[:2] + 'A' + text[3:]
        elif text[2] == '6':
            text = text[:2] + 'G' + text[3:]
        elif text[2] == '0':
            text = text[:2] + 'D' + text[3:]
        elif text[2] == '1':
            text = text[:2] + 'A' + text[3:]
        elif text[2] == '7':
            text = text[:2] + 'A' + text[3:]


    if plate_pattern.match(text):
        return text
    return text


import cv2


def preprocess_plate(plate_img):
    # Chuyển ảnh sang xám
    gray_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Tăng độ tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray_img)


    return contrast_img


def process_image_video(path):
    global tracker
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window.geometry(f"{width}x{height}")
    frame_count = 0
    prev_time = time.time()

    # Initialize SORT tracker
    tracker = Sort()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_plate = False
        boxes = []

        for result in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
            plate_img = frame[y_min:y_max, x_min:x_max]
            plate_img = preprocess_plate(plate_img)
            ocr_result = reader.readtext(plate_img)

            if ocr_result:
                plate_text1 = ' '.join([text[1] for text in ocr_result])
                plate_text = re.sub(r'[^0-9a-zA-Z-]', '', plate_text1)
                plate_text = refine_ocr_text(plate_text)

                if plate_pattern.match(plate_text):
                    confidence = sum([text[2] for text in ocr_result]) / len(ocr_result) * 100
                    label = f"{plate_text} ({confidence:.2f}%)"
                    boxes.append([x_min, y_min, x_max, y_max])  # Append box coordinates for tracking

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    text_width, text_height = text_size
                    text_x1, text_y1 = x_min, y_min - 20
                    text_x2, text_y2 = x_min + text_width, y_min - 20 + text_height
                    if text_x2 > frame.shape[1]:
                        text_x1, text_x2 = frame.shape[1] - text_width - 10, frame.shape[1] - 10
                    if text_x1 < 0:
                        text_x1, text_x2 = 10, 10 + text_width
                    if text_y1 < 0:
                        text_y1, text_y2 = y_min + 20, y_min + 20 + text_height
                    if text_y2 > frame.shape[0]:
                        text_y2, text_y1 = frame.shape[0] - 10, frame.shape[0] - 10 - text_height

                    cv2.rectangle(frame, (text_x1 - 5, text_y1 - 5), (text_x2, text_y2), (255, 255, 255), -1)
                    cv2.putText(frame, label, (text_x1, text_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (11, 176, 116), 2)
                    detected_plate = True

        # Track vehicles across frames using SORT
        if boxes:
            trackers = tracker.update(boxes)
            for d in trackers:
                x1, y1, x2, y2, track_id = map(int, d)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        update_image(frame, img_label)
        window.update_idletasks()
        window.update()

    cap.release()


def process_image(path):

    frame = cv2.imread(path)

    height, width = frame.shape[:2]
    window.geometry(f"{width}x{height}")

    results = model(frame)
    detected_plate = False

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = map(int, result.xyxy[0])
        plate_img = frame[y_min:y_max, x_min:x_max]


        ocr_result = reader.readtext(plate_img)

        if ocr_result:
            plate_text1 = ' '.join([text[1] for text in ocr_result])
            plate_text = re.sub(r'[^0-9a-zA-Z-]', '', plate_text1)
            plate_text = refine_ocr_text(plate_text)
            confidence = sum([text[2] for text in ocr_result]) / len(ocr_result) * 100

            label = f"{plate_text} ({confidence:.2f}%)"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_width, text_height = text_size


            text_x1 = x_min
            text_y1 = y_min - 20
            text_x2 = x_min + text_width
            text_y2 = y_min - 20 + text_height


            if text_x2 > frame.shape[1]:
                text_x1 = frame.shape[1] - text_width - 10
                text_x2 = frame.shape[1] - 10
            if text_x1 < 0:
                text_x1 = 10
                text_x2 = 10 + text_width
            if text_y1 < 0:
                text_y1 = y_min + 20
                text_y2 = text_y1 + text_height
            if text_y2 > frame.shape[0]:
                text_y2 = frame.shape[0] - 10
                text_y1 = text_y2 - text_height

            cv2.rectangle(frame, (text_x1 - 5, text_y1 - 5),
                          (text_x2, text_y2),
                          (255, 255, 255), -1)

            cv2.putText(frame, label, (text_x1, text_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (11, 176, 116), 2)
            detected_plate = True

    if detected_plate:

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join("output", f"plate_{timestamp}.jpg")
        cv2.imwrite(output_filename, frame)


    update_image(frame, img_label)



def select_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("Video files", "*.mp4 *.avi")])

    if file_path:

        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(file_path)
        elif file_path.lower().endswith(('.mp4', '.avi')):
            process_image_video(file_path)



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
    prev_time = time.time()  # Lưu thời gian bắt đầu

    # Tạo label FPS trong giao diện Tkinter, đặt vị trí ở góc trái trên
    fps_label = Label(window, text="FPS: 0.00", font=("Helvetica", 16), fg="red")
    fps_label.place(x=10, y=50)  # Vị trí ở góc trái trên cửa sổ
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
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            if ocr_result:
                plate_text1 = ' '.join([text[1] for text in ocr_result])
                plate_text = re.sub(r'[^0-9a-zA-Z-]', '', plate_text1)
                plate_text = refine_ocr_text(plate_text)

                if plate_pattern.match(plate_text):
                    confidence = sum([text[2] for text in ocr_result]) / len(ocr_result) * 100

                    label = f"{plate_text} ({confidence:.2f}%)"
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    text_width, text_height = text_size


                    text_x1 = x_min
                    text_y1 = y_min - 20
                    text_x2 = x_min + text_width
                    text_y2 = y_min - 20 + text_height


                    if text_x2 > frame.shape[1]:
                        text_x1 = frame.shape[1] - text_width - 10
                        text_x2 = frame.shape[1] - 10
                    if text_x1 < 0:
                        text_x1 = 10
                        text_x2 = 10 + text_width
                    if text_y1 < 0:
                        text_y1 = y_min + 20
                        text_y2 = text_y1 + text_height
                    if text_y2 > frame.shape[0]:
                        text_y2 = frame.shape[0] - 10
                        text_y1 = text_y2 - text_height
                    cv2.rectangle(frame, (text_x1 - 5, text_y1 - 5),
                                  (text_x2, text_y2),
                                  (255, 255, 255), -1)

                    cv2.putText(frame, label, (text_x1, text_y2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (11, 176, 116), 2)
                    detected_plate = True
                else:
                    detected_plate = False
        if detected_plate:
            output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_filename, frame)
        # Tính toán FPS chính xác hơn
        curr_time = time.time()
        frame_time = curr_time - prev_time
        if frame_time > 0:
            fps = 1 / frame_time  # FPS = 1 / thời gian mỗi khung hình

        prev_time = curr_time

        # Cập nhật FPS trong Tkinter
        fps_label.config(text=f"FPS: {fps:.2f}")

        update_image(frame, img_label)
        window.update_idletasks()
        window.update()
        # time.sleep(adjusted_frame_delay)

    cap.release()


# Hàm xử lý phím tắt
def handle_key(event):
    global is_running
    if event.char == 't':
        select_file()
    elif event.char == 'q':
        is_running = False
        window.quit()



window.bind('<KeyPress>', handle_key)



def select_directory():
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_images_in_directory(folder_path)



def process_images_in_directory(directory_path):
    global is_running


    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)


        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4')):
            print(f"Processing image: {filename}")
            process_image_video(file_path)



select_button = Button(window, text="Select Image/Video", command=select_file, bg="#008000", font=("Arial", 18))
select_button.pack(pady=20)

camera_button = Button(window, text="Use Camera", command=use_camera, bg="#008000", font=("Arial", 18))
camera_button.pack(pady=20)


select_directory_button = Button(window, text="Select Folder of Images", command=select_directory, bg="#008000",
                                 font=("Arial", 18))
select_directory_button.pack(pady=20)

window.mainloop()
