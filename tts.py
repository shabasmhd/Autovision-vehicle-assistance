import pyttsx3
import time
import os
import threading


#tts


# File path where the recognized traffic signs are saved
file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"

# Initialize the TTS engine
engine = pyttsx3.init()

# Function to configure the TTS properties (rate, volume)
def configure_tts():
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

# Function to speak a given text
def speak(text):
    engine.say(text)  # Converts the text to speech
    engine.runAndWait()  # Waits for the speech to finish

# Function to monitor the text file and perform TTS
def live_tts():
    print("Starting live TTS... Listening to file updates.")
    configure_tts()  # Configure TTS settings

    # Check if the file exists, if not, create it
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write('')  # Create an empty file if it doesn't exist

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Move the pointer to the end of the file initially
        file.seek(0, os.SEEK_END)

        # Keep track of the last line that was spoken
        last_spoken_sign = ""

        # Loop to continuously monitor the file for changes
        while True:
            try:
                # Read the new sign (if any)
                new_line = file.readline().strip()

                # If a new line is found, and it's different from the last spoken sign
                if new_line and new_line != last_spoken_sign:
                   # print(f"Detected new sign: {new_line}")
                    speak(new_line)  # Speak the new text
                    last_spoken_sign = new_line  # Update the last spoken sign

                # Sleep for a short while to avoid constant CPU usage
                time.sleep(0.2)

            except Exception as e:
                print(f"An error occurred: {e}")
                break

# Start the TTS in a separate thread

tts_thread = threading.Thread(target=live_tts, daemon=True)
tts_thread.start()