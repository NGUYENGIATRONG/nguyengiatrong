import cv2
import numpy as np
import os

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('yolov3.txt', 'r') as f:
    classes = f.read().splitlines()

output_folder = 'Ket_qua'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    # Nhập tên ảnh từ người dùng
    image_path = input("Nhập tên ảnh (hoặc 'exit' để thoát): ")

    if image_path.lower() == 'exit':
        break

    try:
        img = cv2.imread(image_path)

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        net.setInput(blob)

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
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
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))

        if len(indexes) > 0:
            for i in indexes.flatten():Q
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])

                # Tạo thư mục label nếu chưa tồn tại
                label_folder = os.path.join(output_folder, label)
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)

                # Lưu ảnh cắt vào thư mục label
                cropped_img = img[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(label_folder, f'cropped_{i}.jpg'), cropped_img)

        # Hiển thị ảnh gốc với bounding boxes
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Lỗi: {e}")

# Kết thúc chương trình
print("Thoát chương trình.")
