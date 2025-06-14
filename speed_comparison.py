"""

import time
import pyttsx3

def read_latest_value(file_path):
    
    last_value = None
    while True:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    new_value = float(lines[-1].strip())  # Read the latest value
                    if new_value != last_value:  # Check if it's a new value
                        last_value = new_value  # Update last_value
                        return new_value  # Return the updated value
        except ValueError:
            print(f"Invalid data in {file_path}. Ensure it contains only numbers.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        time.sleep(1)  # Avoid busy waiting

def alert_driver():
   
    engine = pyttsx3.init()
    engine.say("Go slow")
    engine.runAndWait()

def monitor_speed(vehicle_speed_file, speed_limit_file):
   02
    
    last_vehicle_speed = None  # Track last read values
    last_speed_limit = None

    while True:
        vehicle_speed = read_latest_value(vehicle_speed_file)  # Read latest vehicle speed
        speed_limit = read_latest_value(speed_limit_file)  # Read latest speed limit

        # Ensure values have been updated
        if vehicle_speed is not None and speed_limit is not None:
            last_vehicle_speed = vehicle_speed  # Update stored values
            last_speed_limit = speed_limit

        # Keep checking if vehicle speed exceeds the speed limit
        while last_vehicle_speed > last_speed_limit:
            alert_driver()
            time.sleep(1)  # Wait before rechecking

            # Update the speed in case it has changed
            last_vehicle_speed = read_latest_value(vehicle_speed_file)

        time.sleep(2)  # Check for new values every 2 seconds

if __name__ == "__main__":
    vehicle_speed_file = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\car_speed.txt"
    speed_limit_file = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\sign text\speed.txt"
    monitor_speed(vehicle_speed_file, speed_limit_file)
"""

import time
import pyttsx3

def read_latest_value(file_path):
    """Reads the last value from a file and returns it."""
    while True:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:
                    return float(lines[-1].strip())  # Always return the last value
        except ValueError:
            print(f"Invalid data in {file_path}. Ensure it contains only numbers.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        time.sleep(1)  # Avoid busy waiting

def alert_driver():
    """Plays the 'Go slow' warning."""
    engine = pyttsx3.init()
    engine.say("Go slow")
    engine.runAndWait()

def monitor_speed(vehicle_speed_file, speed_limit_file):
    """Continuously monitors vehicle speed and alerts if it exceeds the speed limit."""
    
    while True:
        vehicle_speed = read_latest_value(vehicle_speed_file)  # Get latest vehicle speed
        speed_limit = read_latest_value(speed_limit_file)  # Get latest speed limit

        # Continuously check if vehicle speed is above limit
        while vehicle_speed > speed_limit:
            alert_driver()
            time.sleep(1)  # Wait before checking again

            # Update both speed and limit to ensure latest values are used
            vehicle_speed = read_latest_value(vehicle_speed_file)
            speed_limit = read_latest_value(speed_limit_file)

        time.sleep(2)  # Check again after a delay

if __name__ == "__main__":
    vehicle_speed_file = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\car_speed.txt"
    speed_limit_file = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\sign text\speed.txt"
    monitor_speed(vehicle_speed_file, speed_limit_file)
