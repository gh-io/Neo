import pytest
from brain.sensors.lidar import LidarReader

def test_lidar_mock():
    lidar = LidarReader(use_mock=True)
    data = lidar.read()
    assert len(data) > 0
    assert all(isinstance(d, float) for d in data)
