#working pgm of testing

"""
import numpy as np
import cv2 as cv
from keras import models
from skimage import transform, exposure
from ultralytics import YOLO
from time import time, sleep
from PIL import Image
from customtkinter import CTkImage
import os

# IMAGE PREPROCESSING FUNCTION
def image_processing(image):
    image = transform.resize(image, (32, 32))  # Resizing images to 32x32 pixels.
    image = exposure.equalize_adapthist(image,
                                        clip_limit=0.1)  # Applying Histogram Equalization to standardize lighting.
    image = image.astype("float32") / 255.0  # Normalizing image values between 0 and 1.
    return image

def getClassName(classNo):
    classes = ['Speed limit (20km/h)',
               'Speed limit (30km/h)',
               'Speed limit (50km/h)',
               'Speed limit (60km/h)',
               'Speed limit (70km/h)',
               'Speed limit (80km/h)',
               'End of speed limit (80km/h)',
               'Speed limit (100km/h)',
               'Speed limit (120km/h)',
               'No Overtaking',
               'No Overtaking for Heavy Vehicles',
               'Right-of-Way at next Intersection',
               'Priority Road',
               'Yield',
               'Stop',
               'No Vehicles',
               'Heavy Vehicles Prohibited',
               'No Entry',
               'General Caution',
               'Dangerous Left Curve',
               'Dangerous Right Curve',
               'Double Curve',
               'Bumpy Road',
               'Slippery Road',
               'Narrowing Road',
               'Road Work',
               'Traffic Signals',
               'Pedestrian',
               'Children',
               'Bike',
               'Snow',
               'Deer',
               'End of Limits',
               'Turn Right Ahead',
               'Turn Left Ahead',
               'Ahead Only',
               'Go Straight or Right',
               'Go Straight or Left',
               'Keep Right',
               'Keep Left',
               'Roundabout Mandatory',
               'End of No Overtaking',
               'End of No Overtaking for Heavy Vehicles']
    return classes[classNo]

def predictImage(image, model):
    image = image_processing(image)
    image = image.reshape(1, 32, 32, 3)
    prediction = model.predict(image, verbose=0)
    return prediction


def is_speed_already_logged(speed, speed_file_path):
    if os.path.exists(speed_file_path):
        with open(speed_file_path, "r") as sf:
            logged_speeds = sf.read().splitlines()
        return speed in logged_speeds
    return False



"""
"""
# Save the recognized sign name to a file (append mode)
def save_sign_to_file(sign_name):
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
    with open(file_path, "a") as f:
        f.write(f"{sign_name}\n")
"""
"""
import re

# Save the recognized sign name to a file (append mode)
def save_sign_to_file(sign_name):
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
    speed_file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/speed.txt"

    with open(file_path, "a") as f:
        f.write(f"{sign_name}\n")

    # Extract integer speed from sign name
    speed_match = re.search(r'\d+', sign_name)
    if speed_match:
        speed = speed_match.group()
        if not is_speed_already_logged(speed, speed_file_path):
            with open(speed_file_path, "a") as sf:
                sf.write(f"{speed}\n")


# Function to read new lines from a file when new content is appended
def read_new_lines(file_path):
    with open(file_path, "r") as f:
        # Move to the end of the file initially
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline().strip()
            if line:  # New content found
                yield line
            else:
                # Sleep briefly to avoid busy waiting
                sleep(0.5)

# Function to monitor the sign text file for new content
def monitor_sign_file(file_path):
    print("Monitoring file for new signs...")
    for line in read_new_lines(file_path):
        # Save the recognized sign every time it is read
        save_sign_to_file(line)
        print(f"Sign '{line}' written to file again.")

def start_webcam_inference(model_name, condition_prerecorded, camgui=None):
    frameWidth = 640  # CAMERA RESOLUTION
    frameHeight = 640
    brightness = 180
    threshold = 0.5  # PROBABILITY THRESHOLD
    font = cv.FONT_HERSHEY_SIMPLEX

    # SET UP THE VIDEO CAMERA
    if condition_prerecorded:
        print("\nLoading Video...")
        cap = cv.VideoCapture(r"video.mp4")
    else:
        print("\nSetting up Iriun webcam...")
        cap = cv.VideoCapture(1)  # Use the correct index for the Iriun feed
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)
    cap.set(cv.CAP_PROP_FPS, 60)  # Try setting the frame rate to 30 FPS

    # IMPORT THE TRAINED MODEL
    print("\nLoading models from disk...")
    t1 = time()
    yolo_model = YOLO("./runs/detect/train/weights/best.pt")
    print(f"Loaded YOLO model. Took {time() - t1} seconds.")
    t1 = time()
    model = models.load_model('./Models/' + model_name)
    print(f"Loaded CNN model. Took {time() - t1} seconds.")

    print("\nBeginning inference...")
    
    while True:
        # READING IMAGE
        success, imgOriginal = cap.read()
        if not success:
            print("Failed to capture frame. Check Iriun feed.")
            break
        
        if condition_prerecorded:
            imgOriginal = cv.resize(imgOriginal, (640, 480))
        img = np.asarray(imgOriginal)

        # DETECTING SIGN AND EXTRACTING ROI
        results = yolo_model.predict(source=img,conf=0.6, verbose=True)
        img_plotted = results[0].plot()
        x1, y1, x2, y2 = 0, 0, 0, 0
        detected = False
        try:
            boxes = results[0].boxes
            box = boxes[0]  # returns one box
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            detected = True
        except Exception:
            pass

        if detected:
            roi = img[y1:y2, x1:x2]

            # PREDICTING TRAFFIC SIGN CLASS FROM ROI
            predictions = predictImage(roi, model)
            classIndex = np.argmax(predictions)
            probabilityValue = np.amax(predictions)

            # SHOWING RESULTS
            if probabilityValue > threshold:
                recognized_class_name = getClassName(classIndex)
                cv.putText(img_plotted, "CLASS: " + str(classIndex) + " " + str(recognized_class_name), (8, 24),
                           font, 0.50, (0, 0, 255), 1, cv.LINE_AA)
                cv.putText(img_plotted, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (8, 48),
                           font, 0.50, (0, 0, 255), 1, cv.LINE_AA)
                print(f"Detected Traffic Sign.\tClass: {classIndex}\t"
                      f"Probability: {round(probabilityValue * 100, 2):02.2f}%\t\t"
                      f"Name: {recognized_class_name}")

                # Save recognized sign to file every time
                save_sign_to_file(recognized_class_name)

        if camgui is None:
            cv.imshow("Iriun Feed", img_plotted)
        else:
            cvimage = cv.cvtColor(img_plotted, cv.COLOR_BGR2RGBA)
            image = Image.fromarray(cvimage)
            imgctk = CTkImage(image, size=(640, 480))
            camgui.image_label.configure(image=imgctk)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print("Closing Inference Window...")

if __name__ == "__main__":
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
"""
"""
    # Ensure the file exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")  # Create an empty file if it doesn't exist

    # Start monitoring the file for new signs
    try:
        monitor_sign_file(file_path)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
"""
"""
"""
import numpy as np
import cv2 as cv
from keras import models
from skimage import transform, exposure
from ultralytics import YOLO
from time import time, sleep
from PIL import Image
from customtkinter import CTkImage
import os

# IMAGE PREPROCESSING FUNCTION
def image_processing(image):
    image = transform.resize(image, (32, 32))  # Resizing images to 32x32 pixels.
    image = exposure.equalize_adapthist(image,
                                        clip_limit=0.1)  # Applying Histogram Equalization to standardize lighting.
    image = image.astype("float32") / 255.0  # Normalizing image values between 0 and 1.
    return image

def getClassName(classNo):
    classes = ['Speed limit (20km/h)',
               'Speed limit (30km/h)',
               'Speed limit (50km/h)',
               'Speed limit (60km/h)',
               'Speed limit (70km/h)',
               'Speed limit (80km/h)',
               'End of speed limit (80km/h)',
               'Speed limit (100km/h)',
               'Speed limit (120km/h)',
               'No Overtaking',
               'No Overtaking for Heavy Vehicles',
               'Right-of-Way at next Intersection',
               'Priority Road',
               'Yield',
               'Stop',
               'No Vehicles',
               'Heavy Vehicles Prohibited',
               'No Entry',
               'General Caution',
               'Dangerous Left Curve',
               'Dangerous Right Curve',
               'Double Curve',
               'Bumpy Road',
               'Slippery Road',
               'Narrowing Road',
               'Road Work',
               'Traffic Signals',
               'Pedestrian',
               'Children',
               'Bike',
               'Snow',
               'Deer',
               'End of Limits',
               'Turn Right Ahead',
               'Turn Left Ahead',
               'Ahead Only',
               'Go Straight or Right',
               'Go Straight or Left',
               'Keep Right',
               'Keep Left',
               'Roundabout Mandatory',
               'End of No Overtaking',
               'End of No Overtaking for Heavy Vehicles']
    return classes[classNo]

def predictImage(image, model):
    image = image_processing(image)
    image = image.reshape(1, 32, 32, 3)
    prediction = model.predict(image, verbose=0)
    return prediction
"""
# Save the recognized sign name to a file (append mode)
def save_sign_to_file(sign_name):
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
    with open(file_path, "a") as f:
        f.write(f"{sign_name}\n")
"""

import re


# Save the recognized sign name to a file (append mode)
def save_sign_to_file(sign_name):
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
    speed_file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/speed.txt"

    # Append sign name to file
    with open(file_path, "a") as f:
        f.write(f"{sign_name}\n")

    # Extract integer speed from sign name
    speed_match = re.search(r'\d+', sign_name)
    if speed_match:
        speed = speed_match.group()

        # Read existing speeds to prevent duplication
        if os.path.exists(speed_file_path):
            with open(speed_file_path, "r") as sf:
                existing_speeds = sf.read().splitlines()
        else:
            existing_speeds = []

        # Write speed only if it is not already in the file
        if speed not in existing_speeds:
            with open(speed_file_path, "a") as sf:
                sf.write(f"{speed}\n")


# Function to read new lines from a file when new content is appended
def read_new_lines(file_path):
    with open(file_path, "r") as f:
        # Move to the end of the file initially
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline().strip()
            if line:  # New content found
                yield line
            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.5)

# Function to monitor the sign text file for new content
def monitor_sign_file(file_path):
    print("Monitoring file for new signs...")
    for line in read_new_lines(file_path):
        # Save the recognized sign every time it is read
        save_sign_to_file(line)
        print(f"Sign '{line}' written to file again.")



# monitor_sign_file("D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt")

def start_webcam_inference(model_name, condition_prerecorded, camgui=None):
    frameWidth = 640  # CAMERA RESOLUTION
    frameHeight = 640
    brightness = 180
    threshold = 0.5  # PROBABILITY THRESHOLD
    font = cv.FONT_HERSHEY_SIMPLEX

    # SET UP THE VIDEO CAMERA
    if condition_prerecorded:
        print("\nLoading Video...")
        cap = cv.VideoCapture(r"video.mp4")
    else:
        print("\nSetting up Iriun webcam...")
        cap = cv.VideoCapture(0)  
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)
    cap.set(cv.CAP_PROP_FPS, 60)  

    # IMPORT THE TRAINED MODEL
    print("\nLoading models from disk...")
    t1 = time()
    yolo_model = YOLO("./runs/detect/train/weights/best.pt")
    print(f"Loaded YOLO model. Took {time() - t1} seconds.")
    t1 = time()
    model = models.load_model('./Models/' + model_name)
    print(f"Loaded CNN model. Took {time() - t1} seconds.")

    print("\nBeginning inference...")
    
    while True:
        # READING IMAGE
        success, imgOriginal = cap.read()
        if not success:
            print("Failed to capture frame. Check Iriun feed.")
            break
        
        if condition_prerecorded:
            imgOriginal = cv.resize(imgOriginal, (640, 480))
        img = np.asarray(imgOriginal)

        # DETECTING SIGN AND EXTRACTING ROI
        results = yolo_model.predict(source=img,conf=0.6, verbose=True)
        img_plotted = results[0].plot()
        x1, y1, x2, y2 = 0, 0, 0, 0
        detected = False
        try:
            boxes = results[0].boxes
            box = boxes[0]  # returns one box
            x1 = int(box.xyxy[0][0])
            y1 = int(box.xyxy[0][1])
            x2 = int(box.xyxy[0][2])
            y2 = int(box.xyxy[0][3])
            detected = True
        except Exception:
            pass

        if detected:
            roi = img[y1:y2, x1:x2]

            # PREDICTING TRAFFIC SIGN CLASS FROM ROI
            predictions = predictImage(roi, model)
            classIndex = np.argmax(predictions)
            probabilityValue = np.amax(predictions)

            # SHOWING RESULTS
            if probabilityValue > threshold:
                recognized_class_name = getClassName(classIndex)
                cv.putText(img_plotted, "CLASS: " + str(classIndex) + " " + str(recognized_class_name), (8, 24),
                           font, 0.50, (0, 0, 255), 1, cv.LINE_AA)
                cv.putText(img_plotted, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (8, 48),
                           font, 0.50, (0, 0, 255), 1, cv.LINE_AA)
                print(f"Detected Traffic Sign.\tClass: {classIndex}\t"
                      f"Probability: {round(probabilityValue * 100, 2):02.2f}%\t\t"
                      f"Name: {recognized_class_name}")

                # Save recognized sign to file every time
                save_sign_to_file(recognized_class_name)

        if camgui is None:
            cv.imshow("Iriun Feed", img_plotted)
        else:
            cvimage = cv.cvtColor(img_plotted, cv.COLOR_BGR2RGBA)
            image = Image.fromarray(cvimage)
            imgctk = CTkImage(image, size=(640, 480))
            camgui.image_label.configure(image=imgctk)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print("Closing Inference Window...")

if __name__ == "__main__":
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"
"""
    # Ensure the file exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")  # Create an empty file if it doesn't exist

    # Start monitoring the file for new signs
    try:
        monitor_sign_file(file_path)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
"""


