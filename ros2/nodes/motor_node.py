import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

ser = serial.Serial('/dev/ttyUSB0', 115200)

class MotorSubscriber(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.subscription = self.create_subscription(
            String, 'motor_commands', self.listener_callback, 10)

    def listener_callback(self, msg):
        ser.write((msg.data + "\n").encode())

def main(args=None):
    rclpy.init(args=args)
    node = MotorSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    ser.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
