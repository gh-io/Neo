import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import cv2
import serial
import numpy as np
import time
import paho.mqtt.client as mqtt  # swarm communication
