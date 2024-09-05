import streamlit as st
import cv2
import numpy as np
import os
import requests
import re

# URL to download the weights file (Google Drive, Dropbox, etc.)
WEIGHTS_URL = 'https://drive.google.com/file/d/1cvZIX0RiTfl2r-KHSNrzSRNAX6x9h-Uz/view?usp=drive_link'

# Function to download yolov3.weights
def download_yolo_weights():
    weights_path = "yolov3.weights"
    if not os.path.exists(weights_path):
        with st.spinner("Downloading yolov3.weights..."):
            response = requests.get(WEIGHTS_URL, stream=True)
            with open(weights_path, 'wb') as f:
                total_length = response.headers.get('content-length')

                if total_length is None:  # No content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        # Progress bar in console
                        done = int(50 * dl / total_length)
                        st.write(f"Downloading... [{'=' * done}{' ' * (50 - done)}] {done * 2}%")
            st.success("yolov3.weights downloaded!")

# Download the weights if they are not present
download_yolo_weights()

# Load YOLO model
net = cv2.dnn.readNet(weights_path, "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Function to extract price from the label
def extract_price(label):
    match = re.search(r"RM(\d+)", label)
    return int(match.group(1)) if match else 0

# Object detection function
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    total_price = 0
    object_counts = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            price = extract_price(label)
            total_price += price

            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

            # Draw rectangle and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display total price and object counts on the frame
    y_offset = 30
    for obj_label, count in object_counts.items():
        cv2.putText(frame, f"{obj_label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 30

    cv2.putText(frame, f"Total: RM{total_price}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

# Streamlit app UI
st.title("Real-Time Object Detection with YOLO")

# Start webcam capture
run = st.checkbox('Run Object Detection')

# Freeze functionality
freeze = st.checkbox('Freeze Detection')

# Stream video feed in Streamlit
frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not freeze and ret:
        frame = detect_objects(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

cap.release()
