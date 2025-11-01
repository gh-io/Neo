class MockSensors:
    """
    Provides mock sensor data for testing/simulation.
    """

    def read_lidar(self):
        return [1.0, 2.0, 3.0, 4.0]

    def read_camera(self):
        width, height = 640, 480
        return [[0 for _ in range(width)] for _ in range(height)]
