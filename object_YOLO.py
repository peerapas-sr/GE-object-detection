from ultralytics import YOLO
import cv2
import time
import threading
import winsound

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")

# Table ROI (adjust manually)
table_area = (100, 250, 500, 480)

# Shared frame between threads
frame = None
stop_flag = False

# Timer variables
last_human_time = time.time()
alert_delay = 10  # 60 seconds = 1 minute

def capture_frames():
    """Thread for capturing frames from webcam."""
    global frame, stop_flag
    cap = cv2.VideoCapture(0)
    while not stop_flag:
        ret, f = cap.read()
        if not ret:
            break
        frame = f
    cap.release()

# Start camera thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    if frame is None:
        continue  # Wait for first frame

    display_frame = frame.copy()  # Copy frame for drawing

    # Add delay → makes detection slower but smoother video
    time.sleep(0.2)  # 5 FPS detection

    results = model(display_frame, stream=True)

    human_present = False
    object_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                human_present = True
                last_human_time = time.time()  # Reset timer when human is present
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                # Objects only inside table area
                tx1, ty1, tx2, ty2 = table_area
                if x1 >= tx1 and y1 >= ty1 and x2 <= tx2 and y2 <= ty2:
                    object_count += 1
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(display_frame, f"{label} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # Draw table ROI
    tx1, ty1, tx2, ty2 = table_area
    cv2.rectangle(display_frame, (tx1, ty1), (tx2, ty2), (255,255,0), 2)

    # Show counter
    cv2.putText(display_frame, f"Objects on Table: {object_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Calculate time since last human presence
    time_since_human = time.time() - last_human_time

    if object_count > 0 and not human_present:
        # Show countdown timer
        remaining_time = max(0, alert_delay - time_since_human)
        cv2.putText(display_frame, f"Alert in: {int(remaining_time)}s",
                    (10, display_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    
        cv2.putText(display_frame, "NOTICE: Object on table but NO Human!",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Only beep after delay
        if time_since_human >= alert_delay:
            winsound.Beep(1000, 500)

    cv2.imshow("Detection", display_frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Stop camera thread
stop_flag = True
thread.join()
cv2.destroyAllWindows()
