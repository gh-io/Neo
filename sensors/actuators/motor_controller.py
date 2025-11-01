class MotorController:
    """
    Controls motors or actuators.
    """

    def __init__(self, max_speed=100):
        self.max_speed = max_speed
        self.current_speed = 0

    def set_speed(self, speed):
        if speed > self.max_speed:
            speed = self.max_speed
        elif speed < 0:
            speed = 0
        self.current_speed = speed
        print(f"Motor speed set to {self.current_speed}")

    def stop(self):
        self.current_speed = 0
        print("Motor stopped")
