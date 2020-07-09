# Shared-AR-workspace
## Requirement
- [turtlebot](https://github.com/turtlebot/turtlebot)
- [turtlebot_navigation](https://github.com/turtlebot/turtlebot_apps/tree/indigo/turtlebot_navigation)
- [rgbd-pose3d](https://github.com/lmb-freiburg/rgbd-pose3d)
- [ros-sharp(UWP version)](https://github.com/siemens/ros-sharp/wiki)
- [iai-kinect2](https://github.com/code-iai/iai_kinect2)
- [octomap-server](http://wiki.ros.org/octomap_server)
## Branch master: ros package for shared AR workspace
- turtlebot:
  - copy additional launch file and map file from the turtlebot folder into the turtlebot package downloaded from the original repo
  - urdf.launch: robot model description, launch initial environment, launch map server, lanuch move base. PCD file: [https://drive.google.com/drive/folders/1Up5CW8ETfZBM80y1TUtGkX5IkmbclsqD?usp=sharing]
  - rtabmap_mapping_kinect2.launch: localization using kinectv2. Map file:[https://drive.google.com/file/d/1v7DF7QlYqN8SfYvfnjYB_8v3O14w9EBT/view?usp=sharing]
- human detect:
  - copy the script in human_detection folder into the original repo and run using the same command in the original repo
  - RosNode_camera_update.py: setup and publish camera frame and human frame
- ros communication:
  - following the instructions in ROS-Sharp Wiki to install communication package in both ROS and Unity
  - web_server.launch: begin communication between ROS and Hololens
- conver_pcd:
  - pcd_pub.launch: start the whole environment including point-cloud map and original object models
  - load_pcd.launch: implement object transformation, compute perception cost, build octomap
- manipulation:
  - manipulation.launch: compute navigation path, publish command to Hololens
  
## Branch Unity: unity package for shared AR workspace
  This branch includes the objects and related scripts used for object manipulation and message communication in Untiy
