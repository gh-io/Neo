import serial
import time
import random

# ----- SERIAL SETUP -----
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize

# ----- SIMPLE RL FUNCTION -----
def decide_action(distance):
    """
    Simple rule-based placeholder.
    Later, replace with neural network RL model.
    """
    if distance < 20:  # obstacle close
        return random.choice(["LEFT", "RIGHT", "BACKWARD"])
    else:
        return "FORWARD"

# ----- MAIN LOOP -----
try:
    while True:
        if ser.in_waiting:
            sensor_data = ser.readline().decode().strip()
            if sensor_data:
                distance = float(sensor_data)
                action = decide_action(distance)
                print(f"Distance: {distance} cm -> Action: {action}")
                ser.write((action + "\n").encode())
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping Neurobot brain...")
    ser.close()
