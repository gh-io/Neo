---
title: NeomindAI Architecture
linkTitle: Architecture
menu:
  main: {parent: "Platform: Web", weight: 2}
---

<h>1️⃣ Folder Structure</h>
<p
content
  └── en
    └── docs
        ├── index.md               # Landing page
        ├── platform
        │   └── web
        │       ├── basics.md      # Basics & quick start
        │       ├── architecture.md # Detailed architecture
        │       ├── neurobot.md    # ROS2  SLAM Neurobot modules
        │       ├── tutorials.md   # Step-by-step guides
        │       └── visualization.md # Dashboards, graphs, 3D renderings
        ├── advanced-learning:
        │   ├── reinforcement.md
        │   ├── cnn.md
        │   └── snn.md
        └── faq.md
<p>
  
2️⃣ Sidebar & Menu (_index.md / index.md)
  
  title: NeomindAI Docs
linkTitle: Docs
no_list: true
menu:
  main: {weight: 1}


Welcome to **NeomindAI Documentation**! Learn the basics, dive into architecture, follow tutorials, and explore advanced AI concepts.

<p>Quick Navigation

>- [Platform: Web](platform/web/basics.md)
>- [Neurobot Modules & ROS2](platform/web/neurobot.md)
>- [Advanced Learning](advanced-learning/reinforcement.md)
>- [Visualization & Dashboards](platform/web/visualization.md)
>- [FAQ](faq.md)


⸻

>3️⃣ Platform Pages

basics.md (already created, all-in-one)
	•	Quick start on web
	•	Input/decision/motor overview
	•	Example Neurobot simulation

>architecture.md

---
title: NeomindAI Architecture
linkTitle: Architecture
menu:
  main: {parent: "Platform: Web", weight: 2}
---

NeomindAI Architecture Overview

 Core Modules

>| Module | Description |
>|--------|-------------|
>| Input Layer | Receives raw sensor data |
>| Sensory Cortex | Processes vision, audio, tactile |
>| Decision Module | AI-based action selection |
>| Motor Cortex | Executes commands |
>| Memory Module | Short-term & long-term storage |
>| Learning Module | Reinforcement / supervised / SNN |

> Architecture diagram and block flows available on the web interface.


⸻

>neurobot.md

---
title: Neurobot Modules
linkTitle: Neurobot
menu:
  main: {parent: "Platform: Web", weight: 3}
---

 Neurobot & ROS2 Integration

>- **ROS2 nodes** for LiDAR, IMU, camera, motor commands  
>- **Swarm coordination** using MQTT  
>- **SLAM integration** with RTAB-Map  
>- **Main script** connects sensors, ANN/SNN, and actuators  
>
See [Tutorials](tutorials.md) for step-by-step examples.



>tutorials.md

---
title: Tutorials
linkTitle: Tutorials
menu:
  main: {parent: "Platform: Web", weight: 4}
---

Web Tutorials

---
title: NeomindAI Architecture
linkTitle: Architecture
menu:
  main: {parent: "Platform: Web", weight: 2}
---

>- **Neurobot Simulation:** basics & visualization  
>- **RL Training Example:** step-by-step guide  
>- **SNN Reflex Example:** run in browser simulator  

visualization.md

---
title: Visualization
linkTitle: Visualization
menu:
  main: {parent: "Platform: Web", weight: 5}
---

Visualization & Dashboards

>- 3D view of Neurobot  
>- Neural activation heatmaps  
>- Sensor readings in real-time  

Use `<canvas>` or WebGL for interactive dashboards.


⸻

4️⃣ Advanced Learning Pages
	>•	reinforcement.md: RL algorithms, training loops, reward systems
	>•	cnn.md: Image recognition and convolution networks
	>•	snn.md: Spiking Neural Networks, energy-efficient learning

Each page links back to Platform: Web and Tutorials for continuity.

⸻

>5️⃣ FAQ Page
	•	Common questions about NeomindAI setup, simulation, and hardware integration
	•	Links to GitHub repo, pre-built packages, and community forums

