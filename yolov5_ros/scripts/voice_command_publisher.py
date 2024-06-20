#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary modules
import rospy  # ROS Python client library
import speech_recognition as sr  # Speech recognition library
from std_msgs.msg import String  # Standard ROS message type for strings

# Define a class for the Voice Command Publisher
class VoiceCommandPublisher:
    def __init__(self):
        # Initialize a ROS publisher on the '/voice_commands' topic
        self.publisher = rospy.Publisher('/voice_commands', String, queue_size=10)
        # Create an instance of the Recognizer class for recognizing speech
        self.recognizer = sr.Recognizer()
        # Create an instance of the Microphone class to use the default microphone as the audio source
        self.microphone = sr.Microphone()

    def listen(self):
        # Continuously listen for voice commands until the node is shut down
        while not rospy.is_shutdown():
            # Use the microphone as the audio source
            with self.microphone as source:
                rospy.loginfo("Listening for voice commands...")  # Log message to indicate listening state
                # Listen for the first phrase and extract it into audio data
                audio = self.recognizer.listen(source)

            try:
                # Recognize the speech using Google's web service
                command = self.recognizer.recognize_google(audio)
                rospy.loginfo(f"Recognized command: {command}")  # Log the recognized command
                # Publish the recognized command to the '/voice_commands' topic
                self.publisher.publish(command)
            except sr.UnknownValueError:
                # Log an error message if the speech is unintelligible
                rospy.loginfo("Could not understand audio")
            except sr.RequestError as e:
                # Log an error message if there is an issue with the recognition service
                rospy.loginfo(f"Could not request results; {e}")

# Main entry point of the script
if __name__ == "__main__":
    # Initialize the ROS node with the name 'voice_command_publisher'
    rospy.init_node('voice_command_publisher', anonymous=True)
    # Create an instance of the VoiceCommandPublisher class
    voice_command_publisher = VoiceCommandPublisher()
    # Start listening for voice commands
    voice_command_publisher.listen()

