

 **🗂 Neurobot ROS2 + SLAM Module**

```
Neurobot/
├── ros2/
│   ├── launch/
│   │   └── neurobot_slam.launch.py       # Launch SLAM + sensors
│   ├── nodes/
│   │   ├── lidar_node.py                 # LiDAR publisher
│   │   ├── imu_node.py                   # IMU publisher
│   │   ├── camera_node.py                # Camera publisher
│   │   └── motor_node.py                 # Subscribes commands, controls motors
│   ├── maps/
│   │   └── saved_maps/                   # Store generated 3D maps
│   └── swarm_node.py                      # MQTT/ROS2 topic for swarm coordination
```

---

## **1️⃣ ROS2 Launch File** (`ros2/launch/neurobot_slam.launch.py`)

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='neurobot_ros2',
            executable='lidar_node',
            name='lidar_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='imu_node',
            name='imu_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='camera_node',
            name='camera_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='motor_node',
            name='motor_node'
        ),
        Node(
            package='neurobot_ros2',
            executable='swarm_node',
            name='swarm_node'
        ),
    ])
```

---

## **2️⃣ LiDAR Node** (`ros2/nodes/lidar_node.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.publisher = self.create_publisher(LaserScan, 'lidar', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)

    def publish_scan(self):
        msg = LaserScan()
        msg.ranges = np.random.rand(360).tolist()  # Replace with real LiDAR
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## **3️⃣ Motor Node (Arduino Command Subscriber)** (`ros2/nodes/motor_node.py`)

```python
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
```

---

## **4️⃣ Swarm Node** (`ros2/nodes/swarm_node.py`)

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "192.168.1.100"
client = mqtt.Client("neurobot01")
client.connect(MQTT_BROKER)

class SwarmNode(Node):
    def __init__(self):
        super().__init__('swarm_node')
        self.create_subscription(LaserScan, 'lidar', self.lidar_callback, 10)

    def lidar_callback(self, msg):
        # Publish sensor info to swarm
        data = {"lidar": msg.ranges}
        client.publish("neurobot/swarm", json.dumps(data))

def main(args=None):
    rclpy.init(args=args)
    node = SwarmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## **5️⃣ SLAM Integration**

* Use **RTAB-Map ROS2 package** for real-time 3D mapping:

```bash
sudo apt install ros-<ros2-distro>-rtabmap-ros
```

* Connect the LiDAR, camera, and IMU topics to **RTAB-Map node** for mapping and localization.

**Launch Example**:

```bash
ros2 launch rtabmap_ros rtabmap.launch.py \
    rgb_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw \
    scan_topic:=/lidar
```

* Generated maps are stored in `/maps/saved_maps` for swarm sharing.

---

## **6️⃣ Workflow Overview**

```
[LiDAR / Camera / IMU Sensors] ---> ROS2 Nodes ---> SLAM Mapping
                               |
                               v
                          ANN / RL / SNN
                               |
                               v
                         Motor Node / Arduino
                               |
                               v
                        Real-world Movement
                               |
                               v
                         Swarm Node <---> Other Neurobots
```

* Each Neurobot runs **local SLAM** and shares **partial maps** via MQTT or ROS2 topics.
* ANN + RL makes **high-level decisions**, SNN handles **reflexive control**.
* Motors receive commands from **motor_node**, sensors feed **real-time data**, swarm node synchronizes multiple robots.

---
