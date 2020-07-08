#!/usr/bin/env python
import sys
import time
import rospy
import cv2
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from tf import TransformBroadcaster, transformations, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from cv_bridge import CvBridge, CvBridgeError
from threading import Thread, Lock
import math
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from scipy.spatial.transform import Rotation as R
import pickle
import numpy as np

# VALUES YOU MIGHT WANT TO CHANGE
COLOR_IMAGE_TOPIC = "/kinect2/qhd/image_color"  # Ros topic of the undistorted color image
DEPTH_MAP_TOPIC = "/kinect2/qhd/image_depth_rect"  # Ros topic of the depth map warped into the color frame
CAMERA_INFO_TOPIC = '/kinect2/qhd/camera_info'  # ROS topic containing camera calibration K

OPE_DEPTH = 1  # in [1, 5]; Number of stages for the 2D network. Smaller number makes the network fast but less accurate
VPN_TYPE = 'fast'  # in {'fast', 'default'}; which 3D architecture to use
CONF_THRESH = 0.25  # threshold for a keypoint to be considered detected (for visualization)
GPU_ID = 0  # id of gpu device
GPU_MEMORY = None  # in [0.0, 1.0 - eps]; percentage of gpu memory that should be used; None for no limit
# NO CHANGES BEYOND THIS LINE


class KinectDataSubscriber:
    """ Holds the most up to date """
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('listener', anonymous=True) 
        self.camera_sub = rospy.Subscriber('CameraPose_ros',PoseStamped, self.camera_callback, queue_size=1, buff_size=10000000)
        self.pub = rospy.Publisher('human_pose', MarkerArray, queue_size=1)
        # data containers and its mutexes
        self.color_image = None
        self.depth_image = None
        self.position = None
        self.rotation = None
        self.center = None
        self.thetas = None
        self.cam_fixed_frame = None
        self.color_mutex = Lock()
        self.depth_mutex = Lock()
        self.camera_mutex = Lock()
        self.tflistener = TransformListener()

        self.coor_mat = None
        self.line_list = None
        self.points = []

        with open('/home/vcla/Workspace/turtle_ws/src/human_detection/src/human_skele.p', 'rb') as f:
            self.skeles = pickle.load(f)
        
        with open('/home/vcla/Workspace/turtle_ws/src/human_detection/src/human_trans.p', 'rb') as f:
            self.human_trans = pickle.load(f)


    def camera_callback(self, data):
        """ Called whenever depth data is available. """
        self.camera_mutex.acquire()
        self.position = [data.pose.position.x, data.pose.position.z, data.pose.position.y]
        self.rotation = [data.pose.orientation.x, -data.pose.orientation.z, data.pose.orientation.y, data.pose.orientation.w]
        self.camera_mutex.release()

    def cal_quaternion(self, gaze_center):
        theta1 = np.arctan2(gaze_center[2], gaze_center[0])
        theta2 = np.arctan2(gaze_center[1], gaze_center[0])
        # if abs(theta1) >= 3.14:
        #     theta1 = theta1 - 3.14 
        return theta1, theta2

    def person2marker(self, pub, br, camera_frame_id):
        shift = 0
        print('received')
        try:
            (trans_cam, rot_cam) = self.tflistener.lookupTransform('map', camera_frame_id, rospy.Time(0))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        self.cam_fixed_frame = [trans_cam, rot_cam]
        
        br.sendTransform(self.cam_fixed_frame[0],
                            self.cam_fixed_frame[1],
                            rospy.Time.now(),
                            'camera_fixed_frame',
                            'map')

        trans = self.human_trans[0]
        trans = np.array(trans) + np.array([0.5, 1, 0])
        rot = self.human_trans[1]
        br.sendTransform((trans[0], trans[1], trans[2]),
                            rot,
                            rospy.Time.now(),
                            'human_frame_fake',
                            'map')
        br.sendTransform((0, 0, 0),
                            transformations.quaternion_from_euler(0, 1.57/2, 0),
                            rospy.Time.now(),
                            'human_frame_fixed',
                            'human_frame_fake')
        br.sendTransform(self.position, self.rotation, rospy.Time.now(),'human_frame', 'human_frame_fixed')
        br.sendTransform((0, 0, 0),
                            transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'ar_frame',
                            'human_frame_fixed')
        br.sendTransform((0, 0, 0),
                            transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'marker_frame',
                            'map')

        try:
            (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        
        trans, rot = np.array(trans_h), np.array(rot_h)
        r = R.from_quat(rot)      
        skeles = np.array(self.skeles) 
        skeles = r.apply(skeles)
        r = R.from_euler('z', 90, degrees = True)
        skeles = r.apply(skeles) + np.array(trans) #+ np.array([0.5, 0.0, 0])
        ma = MarkerArray()
        h = Header(frame_id='marker_frame')
        line_list = Marker(type=Marker.LINE_LIST, id=0)
        line_list.header = h
        line_list.action = Marker.ADD
        line_list.scale.x = 0.03
        
        for pid in range(0, skeles.shape[0], 2):
            point1 = skeles[pid]
            point2 = skeles[pid+1]
            p0 = Point(x=point1[0],
                    y=point1[1],
                    z=point1[2])
            p1 = Point(x=point2[0],
                    y=point2[1],
                    z=point2[2])
            line_list.points.append(p0)
            line_list.points.append(p1)
        line_list.color.r = 1.0
        line_list.color.g = 0.0
        line_list.color.b = 0.0
        line_list.color.a = 1.0
        ma.markers.append(line_list)
        self.pub.publish(ma)
        print('published')



class CameraCalibSubscriber():
    def __init__(self, camera_info_topic):
        self.subscriber = rospy.Subscriber(camera_info_topic,
                                        CameraInfo, self.camera_callback, queue_size=1)
        self.stop = False
        self.K = None
        self.camera_frame_id = None

    def camera_callback(self, data):
        self.K = np.reshape(np.array(data.K), [3, 3])
        self.camera_frame_id = data.header.frame_id
        self.stop = True

    def wait_for_calib(self):
        try:
            while not self.stop:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down")

        return self.K, self.camera_frame_id


""" Main """
if __name__ == '__main__':

    # Start Node that reads the kinect topic
    data = KinectDataSubscriber()


    pub = rospy.Publisher('human_pose', MarkerArray, queue_size=1)
    br = TransformBroadcaster()

    # loop
    try:
        while not rospy.is_shutdown():
            data.camera_mutex.acquire()
            if (data.position is not None):
                data.camera_mutex.release()
                # run algorithm
                # coords_pred, det_conf = poseNet.detect(color, depth, mask)

                # publish results
                data.person2marker(pub, br, 'kinect2_link')
            else:
                data.camera_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
