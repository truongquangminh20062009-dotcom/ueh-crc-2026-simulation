# Autonomous Navigation for TurtleBot3 - UEH CRC 2026 Simulation Round

Repository containing the ROS 2 autonomous navigation and control package developed for the simulation round of the UEH Creative Robot Contest 2026.

## 👤 Author
* **Trương Quang Minh** (Ho Chi Minh City University of Economics - UEH)

---

## 🎬 Video Demonstration
* **Link Demo Video:** https://drive.google.com/drive/folders/1vK0_xL_4pVuf_yRzfN3_fmv72RVOjkbN?usp=sharing

---

## 🛠️ Technical Stack
* **OS / Framework:** Ubuntu 22.04, ROS 2 Humble
* **Simulator:** Gazebo Classic 11 / TurtleBot3 Waffle
* **Perception:** OpenCV (HSV color filtering, contour detection, ROI analysis), 360-degree LIDAR (`/scan`)
* **Control Architecture:** Finite State Machine (FSM) combined with a Proportional-Derivative (PD) steering controller.

---

## Repository Structure
```text
ueh-crc-2026-simulation/
├── src/
│   └── crc_sim/
│       ├── config/         # Configuration parameters
│       ├── launch/         # ROS 2 launch files
│       ├── models/         # Track and environment models
│       ├── worlds/         # Gazebo world files
│       ├── crc_sim/        # Python source nodes (starter_node.py, etc.)
│       ├── package.xml     # Package metadata
│       └── setup.py        # Python package build configurations
├── docs/
│   └── Technical_Report.pdf # Final academic-style technical report
└── README.md
