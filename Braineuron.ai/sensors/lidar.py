class LidarReader:
    """
    Reads LIDAR data and outputs distances.

    Args:
        port (str): Serial port.
        max_distance (float): Maximum measurable distance in meters.
        use_mock (bool): Use mock sensor data if True.
    """

    def __init__(self, port="/dev/ttyUSB0", max_distance=10.0, use_mock=False):
        self.port = port
        self.max_distance = max_distance
        self.use_mock = use_mock

    def read(self):
        if self.use_mock:
            return [1.0, 2.0, 3.0]  # dummy values
        # TODO: Add real sensor code
        return []
