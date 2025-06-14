import time
import random

def simulate_realistic_car_speed(output_file=r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\car_speed.txt"):
    """ Simulates vehicle speed with larger variations at 2-second intervals. """
    
    speed = random.randint(30, 80)  
    max_speed = 80 
    min_speed = 30  

    with open(output_file, "w") as file:
        while True:
            change_type = random.choice(["increase", "decrease"])  
            
            if change_type == "increase":
                speed += random.randint(10, 20)  # Big acceleration
            else:
                speed -= random.randint(10, 20)  # Big deceleration
            
            speed = max(min_speed, min(speed, max_speed))

            file.write(f"{speed}\n")
            file.flush()  # Ensure data is written immediately

            time.sleep(2)  

simulate_realistic_car_speed()



