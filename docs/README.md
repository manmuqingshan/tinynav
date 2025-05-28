Welcome to the docs for tinyNav.


## Overview

**TinyNav** is a compact, efficient, and modular navigation framework designed primarily for use with [Intel RealSense depth cameras](https://www.intelrealsense.com/depth-camera-d435/) and NVIDIA [Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/) platforms. It provides essential capabilities for stereo-based visual odometry, mapping, and path planning in real-time robotics applications.
  
![data flow](/docs/rosgraph.svg)

## Module introduce

### Stereo Visual Odometry Module

The odometry module is expected to use pairs of images for relative pose estimation through feature extraction and feature matching procedures.

| Function                          | Description                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| **Depth Estimation**          | Computes dense depth maps from stereo disparity images using **[Semi-Global Block Matching](https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html) (SGBM)**, enabling 3D reconstruction and localization.                 |
| **Feature Extraction & Matching** | Detects keypoints with **[SuperPoint](https://github.com/rpautrat/SuperPoint)** and matches them using **[LightGlue](https://github.com/cvg/LightGlue)**.          |
| **Pose Estimation**               | Estimates relative camera pose via 3D-2D correspondences using **[PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)** methods.      |
| **Odometry Node Publishing(Optional)**           | Publishes estimated pose as `nav_msgs/Odometry` and broadcasts the corresponding TF. |

### Planning Module

The planning module is responsible for generating optimal and feasible control commands for the robot, utilizing information from local perception.

| Function                            | Description                                                                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local Dynamic Mapping**           | Generates a local occupancy grid map from accumulated depth frames and robot poses in real-time.                                              |
| **Trajectory Simulation & Scoring** | Uses **[Dynamic Window Approach (DWA)](https://en.wikipedia.org/wiki/Dynamic_window_approach)** to simulate multiple candidate trajectories, scoring them based on goal alignment, obstacle avoidance. |
| **Optimal Command Selection**       | Selects the best trajectory and computes the corresponding velocity commands for execution.                                                   |
| **Planned Command Node Publishing(optional)**       | Publishes planned command as `geometry_msgs/Twist` and local map as `nav_msgs/OccupancyGrid`.|

### Mapping  Module

The mapping module is responsible for providing the robot’s global position along with static map information.

| Function                  | Description                                                                                                                                                       |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Global Feature Extraction** | Extracts global image descriptors using **[DINO V2 embeddings](https://huggingface.co/docs/transformers/model_doc/dinov2)** for scene recognition, loop detection, and global similarity assessment.                               |
| **Local Bundle Adjustment**   | Performs local optimization of camera poses and 3D landmark positions using a **[Bundle Adjustment](http://ceres-solver.org/nnls_tutorial.html#bundle-adjustment)** solver while respecting relative pose constraints.             |
| **Loop Closure**              | Identifies potential loop closure candidates via **DINO feature similarity**, refines alignment through [pose graph optimization](https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_2d/pose_graph_2d.cc), and updates the map via BA.       |
| **Map Node Publishing (Optional)**      | Publishes the path as `nav_msgs/Path`.                                                                |
| **Global Planning (Optional)**      | Can be extended to include global path planning using cost maps and [**A\* method**](https://en.wikipedia.org/wiki/A*_search_algorithm).                                                                |

