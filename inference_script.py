import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import time
import os
import glob

model_path = "runs/detect/train/weights/best.pt"

test_images_dir = "D:/yolov_11_custom4/test/images"

def simulate_stacklamp_yolo(test_images_dir=test_images_dir, model_path=model_path):
    # Verify model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Check 'runs/detect/' for the correct training run folder (e.g., train5).")
        return None

    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(test_images_dir, ext)))

    if not image_files:
        print(f"Error: No images found in {test_images_dir}")
        return None

    data = {
        'timestamp': [],
        'current_state': [],
        'machinery_status': [],
        'alert': []
    }

    state_map = {0: 'Red', 1: 'Orange', 2: 'Green', 3: 'Person'}
    machinery_map = {0: 'stopped', 1: 'faulty', 2: 'operational'}

    for image_path in image_files:

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            continue

        try:
            results = model(frame, conf=0.3)
        except Exception as e:
            print(f"Error during inference on {image_path}: {str(e)}")
            continue

        current_state = 0
        alert = 0
        person_detected = False

        detected_classes = []
        detection_details = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                detected_classes.append(cls)
                detection_details.append(f"{state_map[cls]} (conf={conf:.2f})")
                if cls in [0, 1, 2]:
                    current_state = cls
                elif cls == 3:
                    person_detected = True

        print(f"Image {os.path.basename(image_path)}: Detected classes = [{', '.join(detection_details)}]")

        machinery_status = current_state

        if current_state == 2 and person_detected:
            alert = 1
            print(f"PERSON PRESENT ALERT!! at {datetime.now()} for image {os.path.basename(image_path)}")
        elif current_state == 1:
            alert = 1
            print(f"NEED A TECHNICIAN FAULT IN MACHINERY!! at {datetime.now()} for image {os.path.basename(image_path)}")


        data['timestamp'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        data['current_state'].append(current_state)
        data['machinery_status'].append(machinery_status)
        data['alert'].append(alert)

        annotated_frame = results[0].plot()  # YOLO provides annotated frame
        cv2.imshow('Stacklamp Detection', annotated_frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):  # Display each image for 500ms
            break

        time.sleep(0.1)

    cv2.destroyAllWindows()

    try:
        df = pd.DataFrame(data)
        df.to_csv('stacklamp_data.csv', index=False)
        print("Data saved to stacklamp_data.csv")
        return df
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
        return None

if __name__ == "__main__":
    df = simulate_stacklamp_yolo()