import sys
import cv2
import os
import contextlib
import time
from ultralytics import YOLO

os.environ['YOLO_VERBOSE'] = '0'

# Get command-line arguments
if len(sys.argv) > 1:
    VIDEO_FILE = sys.argv[1] if sys.argv[1] != "cam" else None
else:
    VIDEO_FILE = None  # If no video file given, takes input from webcam

USE_WEBCAM = VIDEO_FILE is None  

# Initialize video capture
cap = cv2.VideoCapture(0 if USE_WEBCAM else VIDEO_FILE)
if not cap.isOpened():
    sys.exit()

# Continuously open the webcam (keep it "always on")
cap.set(cv2.CAP_PROP_FPS, 30)  # Set a stable FPS to ensure the camera feed is continuous
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# Media player window size
resize_width = 640  
resize_height = 480  
if frame_width > 0:
    resize_height = int((resize_width / frame_width) * frame_height)

skip_frames = 2  
frame_count = 0

# Load YOLO model
chosen_model = YOLO("yolov8n.pt")  

pedestrian_detected_time = 0
pedestrian_detected = False
light_status = "Red" 
no_pedestrian_time = 0  
signal_cnt = 0

def predict(chosen_model, img, classes=[], conf=0.7):
    with contextlib.redirect_stdout(None):  
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False) if classes else chosen_model.predict(img, conf=conf, verbose=False)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    global pedestrian_detected_time, pedestrian_detected, light_status, no_pedestrian_time, signal_cnt

    pedestrian_detected_this_frame = False  # Flag to check if a pedestrian is detected in the current frame

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = box.conf[0] * 100  # Convert to percentage
            
            if class_name == "person":
                class_name = "pedestrian"  # Rename person to pedestrian

            # Choose box color based on pedestrian detection
            color = (0, 255, 0) if class_name == "pedestrian" else (0, 0, 255)

            # Draw the bounding box
            cv2.rectangle(img, 
                          (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                          color, 2)

            # Display label with confidence
            label = f"{class_name}: {confidence:.2f}%"
            cv2.putText(img, label, 
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

            # Detect pedestrian and track time
            if class_name == "pedestrian":
                pedestrian_detected = True
                pedestrian_detected_this_frame = True  # Pedestrian detected in this frame
            else:
                pedestrian_detected = False

    # Update pedestrian_detected_time and no_pedestrian_time
    if pedestrian_detected_this_frame:
        pedestrian_detected_time += 1
        no_pedestrian_time = 0
    else:
        pedestrian_detected_time = max(0, pedestrian_detected_time - 1)
        no_pedestrian_time += 1

    # Update light status and signal_cnt logic
    if pedestrian_detected_time > 150:
        if light_status == "Red":
            light_status = "Yellow"
            pedestrian_detected_time = 0
            signal_cnt = 1  # Set signal_cnt to 1 when light turns Yellow
    if pedestrian_detected_time > 30:
        if light_status == "Yellow":
            light_status = "Green"

    if no_pedestrian_time > 20:
        light_status = "Red"
        signal_cnt = 0  # Reset signal_cnt to 0 when light turns Red

    return img, results, signal_cnt

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # Increment frame_count and skip frames for better performance
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    # Resize image to reduce processing time
    img = cv2.resize(img, (resize_width, resize_height))

    # Process the image and get results
    result_img, _, signal_cnt = predict_and_detect(chosen_model, img, classes=[], conf=0.5)

    # Display light status on the image
    cv2.putText(result_img, f"Light: {light_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the detection result in a window
    cv2.imshow("Detection", result_img)

    # Print the timer (pedestrian_detected_time), signal light (light_status), and signal count (signal_cnt)
    print(f"Pedestrian Detected Time: {pedestrian_detected_time}, Light Status: {light_status}, Signal_cnt: {signal_cnt}")
  
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
