brain = StrategicANN()
reflex_snn = ReflexSNN()
optimizer = optim.Adam(brain.parameters(), lr=0.001)
criterion = nn.MSELoss()

actions = ["FORWARD", "LEFT", "RIGHT", "BACKWARD"]

try:
    cap = cv2.VideoCapture(0)
    while True:
        # ----- Read Arduino Sensors -----
        if ser.in_waiting:
            data = ser.readline().decode().strip()
            if data:
                # Example format: distance,imu_angle, lidar1,lidar2,...,lidar360
                parts = list(map(float, data.split(',')))
                distance = parts[0]
                imu = parts[1]
                lidar = parts[2:]
                sensor_input = process_lidar(lidar + [distance])  # 361 features
        
        # ----- Read Camera -----
        ret, frame = cap.read()
        if not ret:
            continue
        image_input = preprocess_camera(frame)
        
        # ----- ANN Strategic Decision -----
        ann_output = brain(image_input, sensor_input)
        action_idx = torch.argmax(ann_output).item()
        action = actions[action_idx]
        
        # ----- Reflexive SNN Control -----
        reflex_output = reflex_snn(sensor_input)
        # Convert to motor commands (0-180 degrees or speed)
        motor_command = reflex_output.detach().numpy()
        
        # ----- Send commands to Arduino -----
        ser.write((action + "\n").encode())
        
        # ----- Reward + Learning (placeholder for DQN/PPO) -----
        reward = 1 if distance > 15 else -1
        target = torch.zeros_like(ann_output)
        target[0, action_idx] = reward
        optimizer.zero_grad()
        loss = criterion(ann_output, target)
        loss.backward()
        optimizer.step()
        
        # ----- Swarm Communication -----
        position = [0,0,0]  # Example, replace with real odometry
        obstacles = lidar
        publish_state(position, obstacles)
        
        print(f"Action: {action} | Reward: {reward} | Motor spikes: {motor_command}")

except KeyboardInterrupt:
    cap.release()
    ser.close()
    print("Neurobot shut down")
