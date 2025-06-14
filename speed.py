import time
import os

# Mock function to simulate getting the car's speed from an OBD-II device
def get_car_speed():
    import random
    return random.randint(0, 120)

def extract_speed_from_sign(sign_name):
    # Parse the speed from the detected sign name
    try:
        if "Speed limit" in sign_name:
            speed = int(sign_name.split("(")[1].split("km/h")[0])
            return speed
    except Exception as e:
        print(f"Error parsing speed from sign: {e}")
    return None

# Function to compare car speed with detected speed limit
def compare_speed(car_speed, speed_limit):
    if speed_limit is None:
        print("No valid speed limit detected.")
        return

    if car_speed > speed_limit:
        print(f"WARNING: Overspeeding! Car speed: {car_speed} km/h, Speed limit: {speed_limit} km/h")
    else:
        print(f"Safe driving. Car speed: {car_speed} km/h, Speed limit: {speed_limit} km/h")

# Function to read the latest detected speed limit from a file
def read_latest_speed_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            print("Sign text file not found.")
            return None

        with open(file_path, "r") as file:
            lines = file.readlines()
            if lines:
                # Get the latest detected sign
                latest_sign = lines[-1].strip()
                return latest_sign
    except Exception as e:
        print(f"Error reading sign text file: {e}")
    return None

if __name__ == "__main__":
    file_path = "D:/watsapp/Traffic-Sign-Recognition-master/Traffic-Sign-Recognition-master/sign text/signs.txt"

    # Real-time speed comparison
    print("Starting real-time speed comparison...")
    while True:
        # Get current car speed
        car_speed = get_car_speed()

        # Get the latest detected sign from the file
        detected_sign = read_latest_speed_from_file(file_path)
        if detected_sign:
            # Extract speed limit from the detected sign
            speed_limit = extract_speed_from_sign(detected_sign)

            # Compare car speed with detected speed limit
            compare_speed(car_speed, speed_limit)
        else:
            print("No detected signs found in the file.")

        # Wait for a second before next comparison
        time.sleep(1)

