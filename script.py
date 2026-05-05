import RPi.GPIO as GPIO
import cv2
import mediapipe as mp
import numpy as np
import freenect
import threading
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MAX_NUM_PEOPLE = 5 # max number of people to track. 5 is kinda excessive, would go 1<=x<=5
OUTPUT_PIN = 27 # leave this at 27. determines GPIO output pin on the raspberry pi
KNEE_ANGLE_THRESHOLD = 160 # angle threshold to tamper with. standing straight upright is 180. knee half bent is 90. so do like in range 100 to 170.
INHALE_DURATION = 4.0 # seconds to spend inhaling. can be a decimal
EXHALE_DURATION = 8.0 # seconds to spend exhaling. can be a decimal
STALL_DURATION = 0.1 # seconds to spend stalling. can be a decimal. the "stall" time is what happens when we're waiting to see if we should inhale/exhale again. should be relatively low, but not too low. too low results in inefficiencies. 0.1 is a good number.




### DO NOT EDIT BELOW THIS LINE
### ---------------------------------------------------------------














base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=MAX_NUM_PEOPLE
)

detector = vision.PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)


frame_id = 0
# is_breathing = True
# is_breathing_lock = threading.Lock()

GPIO.setmode(GPIO.BCM)
GPIO.setup(OUTPUT_PIN, GPIO.OUT)

GPIO.output(OUTPUT_PIN, GPIO.LOW)

# def breathe():
#     global is_breathing
#     was_breathing = False
#     inhale = True

#     while True:
#         local_breathing = False
#         with is_breathing_lock:
#             local_breathing = is_breathing

#         if local_breathing:
#             if not was_breathing:
#                 inhale = True
#             if inhale:
#                 print("Inhaling")
#                 GPIO.output(OUTPUT_PIN, GPIO.HIGH)
#                 time.sleep(INHALE_DURATION)
#                 inhale = False
#             else:
#                 print("Exhaling")
#                 GPIO.output(OUTPUT_PIN, GPIO.LOW)
#                 time.sleep(EXHALE_DURATION)
#                 inhale = True

#             was_breathing = True
#         else:
#             GPIO.output(OUTPUT_PIN, GPIO.LOW)
#             was_breathing = False
#             print(".")
#             time.sleep(STALL_DURATION)


# t = threading.Thread(target=breathe)
# t.start()


def get_vid():
    frame = freenect.sync_get_video()[0]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_depth():
    return freenect.sync_get_depth()[0]

def compute_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


mp_pose = mp.solutions.pose
try:
    while True:
        GPIO.output(OUTPUT_PIN, GPIO.LOW)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(10, GPIO.LOW)
        rgb = get_vid()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, frame_id)
        frame_id += 1
        #rgb = get_vid()
        #depth = get_depth()

        #results = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        turn_on_pin = False
        if result.pose_landmarks:
            
            for person_id, pose in enumerate(result.pose_landmarks):
                h, w, _ = rgb.shape
                # lm = results.pose_landmarks.landmark
                def get_point(idx):
                    return (
                            pose[idx].x * w,
                            pose[idx].y * h
                            )

                LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
                LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
                LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value

                RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
                RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
                RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

                left_angle = compute_angle(
                    get_point(LEFT_HIP),
                    get_point(LEFT_KNEE),
                    get_point(LEFT_ANKLE)
                
                )

                right_angle = compute_angle(
                    get_point(RIGHT_HIP),
                    get_point(RIGHT_KNEE),
                    get_point(RIGHT_ANKLE)
                )

                if (left_angle < KNEE_ANGLE_THRESHOLD or right_angle < KNEE_ANGLE_THRESHOLD):
                    
                    GPIO.output(OUTPUT_PIN, GPIO.HIGH)
            
                    time.sleep(EXHALE_DURATION)

                    GPIO.output(OUTPUT_PIN, GPIO.LOW)

                    time.sleep(INHALE_DURATION)

                    # turn_on_pin = True
                    # print(f"P{person_id} enables")
            
        else:
            print("NO POSE")

        # if turn_on_pin:
        #     with is_breathing_lock:
        #         is_breathing = True
        #     #print("ON=======================")
            
        # else:
        #     with is_breathing_lock:
        #         is_breathing = False
            #print("off-------")
        
finally:
    GPIO.cleanup()
