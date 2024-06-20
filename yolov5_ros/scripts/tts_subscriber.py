#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary modules
import rospy
from gtts import gTTS  # Google Text-to-Speech
from playsound import playsound  # Play the saved speech audio
import os  # Operating system library for file operations
from std_msgs.msg import String  # Standard ROS message type for strings

# Define a class for the Text-to-Speech subscriber
class TTS_Subscriber:
    def __init__(self):
        # Initialize the subscriber to listen to the '/detection_result' topic
        self.subscriber = rospy.Subscriber('/detection_result', String, self.text_to_speech)

    def text_to_speech(self, msg):
        # Convert the incoming message data to speech
        tts = gTTS(text=msg.data, lang='en')
        # Save the generated speech to a temporary file
        tts.save("/tmp/speech.mp3")
        # Play the saved speech audio file
        playsound("/tmp/speech.mp3")
        # Remove the temporary speech audio file after playing it
        os.remove("/tmp/speech.mp3")

# Main entry point of the script
if __name__ == "__main__":
    # Initialize the ROS node with the name 'tts_subscriber'
    rospy.init_node('tts_subscriber', anonymous=True)
    # Create an instance of the TTS_Subscriber class
    tts_subscriber = TTS_Subscriber()
    # Keep the script running and listening for messages
    rospy.spin()

