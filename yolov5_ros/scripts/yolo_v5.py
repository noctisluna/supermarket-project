#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2  # OpenCV for image processing
import torch  # PyTorch for deep learning models
import rospy  # ROS Python client library
import numpy as np  # Numpy for numerical operations
from gtts import gTTS  # Google Text-to-Speech
from playsound import playsound  # Play audio files
import os  # OS module for file operations
import time  # Time module for time-related operations

from std_msgs.msg import Header, String  # Standard ROS message types
from sensor_msgs.msg import Image  # ROS Image message type
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes  # Custom YOLOv5 message types

class Yolo_Dect:
    def __init__(self):
        self.active = False  # To track whether the node is active
        self.person_detected_once = False  # Flag to track if a person has been detected at least once
        rospy.Subscriber('/voice_commands', String, self.voice_command_callback)  # Subscribe to voice commands

        # Load parameters from ROS parameter server
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # Load YOLOv5 model from local repository
        self.model = torch.hub.load(yolov5_path, 'custom', path=weight_path, source='local')

        # Select device (CPU or GPU)
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf  # Set model confidence threshold
        self.color_image = None  # Placeholder for the color image
        self.getImageStatus = False  # Flag to track if an image is received

        # Dictionary to store class colors for bounding boxes
        self.classes_colors = {}

        # Subscribe to image topic
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)

        # Publishers for bounding boxes and detection images
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=1)
        self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=1)

        # Publisher for DetectionInfo
        self.highest_class_pub = rospy.Publisher('/detection_result', String, queue_size=1)

        self.reported_classes = set()  # Set to track reported classes
        self.new_detection = False  # Flag to indicate a new detection

        # Set up a timer to perform detection every 3 seconds
        self.detection_timer = rospy.Timer(rospy.Duration(3), self.timer_callback)

    def voice_command_callback(self, msg):
        command = msg.data.lower()  # Convert command to lowercase
        if command == "start":
            rospy.loginfo("Starting YOLO detection.")
            self.active = True  # Activate detection
        elif command == "stop":
            rospy.loginfo("Stopping YOLO detection.")
            self.active = False  # Deactivate detection
            rospy.signal_shutdown("Stopping YOLO detection.")  # Shutdown the node

    def image_callback(self, image):
        self.getImageStatus = True  # Image has been received
        # Convert image data to numpy array and reshape
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        # Convert image from BGR to RGB
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

    def timer_callback(self, event):
        # Return if node is not active or no image is available
        if not self.active or self.color_image is None or not self.color_image.any():
            return

        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header.stamp = rospy.Time.now()

        results = self.model(self.color_image)  # Perform detection
        boxs = results.pandas().xyxy[0].values  # Extract bounding box data
        self.dectshow(self.color_image, boxs)  # Display and process detections

        cv2.waitKey(3)  # Wait for a key press (required for OpenCV window updates)

    def dectshow(self, org_img, boxs):
        img = org_img.copy()  # Create a copy of the original image
        reported_detections = set()
        detected_person = False
        highest_prob_class = None
        highest_prob = 0

        count = len(boxs)  # Number of detected boxes
        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])  # Detection confidence
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)  # Number of detections
            boundingBox.Class = box[-1]  # Class label

            # Assign color to class if not already assigned
            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            # Draw bounding box
            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]), int(color[1]), int(color[2])), 2)

            # Position text label appropriately
            text_pos_y = box[1] + 30 if box[1] < 20 else box[1] - 10
            cv2.putText(img, box[-1],
                        (int(box[0]), int(text_pos_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Append bounding box to the message and publish
            self.boundingBoxes.bounding_boxes.append(boundingBox)
            self.position_pub.publish(self.boundingBoxes)

            if boundingBox.Class == 'person':
                detected_person = True

            if boundingBox.Class != 'person' and boundingBox.probability > highest_prob:
                highest_prob = boundingBox.probability
                highest_prob_class = boundingBox.Class

        if detected_person:
            self.person_detected_once = True

        if not detected_person and not self.person_detected_once:
            rospy.loginfo("Please get closer to the robot to detect the goods")
            self.text_to_speech("Please get closer to the robot to detect the goods")
        elif highest_prob_class and highest_prob_class not in self.reported_classes:
            self.new_detection = True
            self.highest_class_pub.publish(highest_prob_class)
            self.reported_classes.add(highest_prob_class)

        # Prices dictionary for detected items
        prices = {
            "cell phone": 1000,
            "laptop": 2000,
            "bottle": 2
        }

        if self.new_detection and highest_prob_class:
            # Get price for the detected item
            price = prices.get(highest_prob_class, "an unknown price")
            # Create message for the detected item
            message = f"{highest_prob_class}, its price is RM{price}, thank you for your purchase."
            rospy.loginfo(message)
            self.text_to_speech(message)  # Convert message to speech
            self.new_detection = False

        self.publish_image(img)  # Publish the processed image
        cv2.imshow('YOLOv5', img)  # Display the image with detections

    def publish_image(self, imgdata):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = imgdata.shape[0]
        image_temp.width = imgdata.shape[1]
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = imgdata.shape[1] * 3
        self.image_pub.publish(image_temp)  # Publish the image to the specified topic

    def text_to_speech(self, text):
        if not text:
            return
        tts = gTTS(text=text, lang='en')  # Convert text to speech
        # Create a unique filename using the current timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_path = f"/tmp/speech_{timestamp}.mp3"
        tts.save(file_path)  # Save the speech to a file

        # Ensure the file is written
        while not os.path.exists(file_path):
            time.sleep(0.1)

        # Play the sound once
        try:
            playsound(file_path)
        except Exception as e:
            rospy.logerr(f"Error playing sound: {e}")

        # Remove the file if it exists
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            rospy.logerr(f"Error removing sound file: {e}")

    def cleanup(self):
        # Stop the timer
        self.detection_timer.shutdown()
        # Unsubscribe from topics
        self.color_sub.unregister()
        # Close any open windows
        cv2.destroyAllWindows()
        rospy.loginfo("Cleaned up resources")

if __name__ == "__main__":
    rospy.init_node('yolov5_ros', anonymous=True)  # Initialize ROS node
    yolo_dect = Yolo_Dect()  # Create YOLO detection object
    rospy.on_shutdown(yolo_dect.cleanup)  # Register cleanup function
    rospy.spin()  # Keep the node running

