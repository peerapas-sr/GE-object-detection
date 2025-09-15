import cv2
import numpy as np

# Load pre-trained MobileNet SSD (COCO dataset)
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Define "table area" as bottom 1/3 of the screen
    table_y_min = int(h * 0.65)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    object_count = 0
    human_present = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Check if it's a person
            if label == "person":
                human_present = True
                color = (0, 255, 0)  # green
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}",
                            (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                # Object must be inside the "table area"
                if startY >= table_y_min:
                    object_count += 1
                    color = (0, 0, 255)  # red
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}",
                                (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw table area rectangle for reference
    cv2.line(frame, (0, table_y_min), (w, table_y_min), (255, 255, 0), 2)
    cv2.putText(frame, "Table Area", (10, table_y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show counts
    cv2.putText(frame, f"Objects on Table: {object_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)

    if object_count > 0 and not human_present:
        cv2.putText(frame, "NOTICE: Object on table but NO Human!",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    cv2.imshow("Detection", frame)

    # Exit with Q or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
