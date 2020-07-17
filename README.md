# Shared-AR-workspace

## 1. Dependencies & Installation

Several dependencies need to be installed before using our EddieBot ROS packages. We list them below:


- [turtlebot](https://github.com/turtlebot/turtlebot)
- [turtlebot_navigation](https://github.com/turtlebot/turtlebot_apps/tree/indigo/turtlebot_navigation)
- [rgbd-pose3d](https://github.com/lmb-freiburg/rgbd-pose3d)
- [ros-sharp(UWP version)](https://github.com/siemens/ros-sharp/wiki)
- [iai-kinect2](https://github.com/code-iai/iai_kinect2)
- [octomap-server](http://wiki.ros.org/octomap_server)


## 2. Usage

In order to use our shared AR workspace, both ends (a ROS system and a HoloLens system) need to be setup according to the instruction below.

### 2.1 ROS End

Download and install the [master branch](https://github.com/Janetalready/Shared-AR-workspace/tree/master) on your desktop that runs ROS. 

- turtlebot:
  - copy additional launch file and map file from the turtlebot folder into the turtlebot package downloaded from the original repo
  - `roslaunch turtlebot_bringup urdf.launch`: lanuch robot model description, launch initial environment, launch map server, lanuch move base. 
  - `roslaunch turtlebot_navigation rtabmap_mapping_kinect2.launch`: localization using kinectv2. Map file:[https://drive.google.com/file/d/1v7DF7QlYqN8SfYvfnjYB_8v3O14w9EBT/view?usp=sharing]. Note: remember to change the map path in the launch file
- human detect:
  - copy the script in human_detection folder into the original repo
  - `python RosNode_camera_update.py`: setup and publish camera frame and human frame
- ros communication:
  - following the instructions in ROS-Sharp Wiki to install communication package in both ROS and Unity
  - `roslaunch ros_communication web_server.launch`: begin communication between ROS and Hololens. Note: please launch after the Hologram is started
- convert_pcd:
  - pcd_pub.launch(included in urdr.lanuch): start the whole environment including point-cloud map and original object models. PCD file: [https://drive.google.com/drive/folders/1Up5CW8ETfZBM80y1TUtGkX5IkmbclsqD?usp=sharing]. Note: remember to modify the pcd path in pcd_pub.launch
  - `roslaunch covert_pcd load_pcd.launch`: implement object transformation, compute perception cost, build octomap
- manipulation:
  - `roslaunch manipulation manipulation.launch`: compute navigation path, publish command to Hololens


### 2.2 HoloLens End

Download the [Unity branch](https://github.com/Janetalready/Shared-AR-workspace/tree/Unity) first, and import the project using [Unity](https://store.unity.com/#plans-individual). Then compile the project, and upload the hologram to your HoloLens.

## Citation

```tex
@inproceedings{qiu2020humanrobot,
  title={Human-Robot Interaction in a Shared Augmented Reality Workspace},
  author={Qiu, Shuwen and Liu, Hangxin and Zhang, Zeyu and Zhu, Yixin and Zhu, Song-Chun},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE}
}
```
