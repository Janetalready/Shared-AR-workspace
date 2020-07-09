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
from PoseNet3D import *
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from scipy.spatial.transform import Rotation as R

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
        self.color_sub = rospy.Subscriber(COLOR_IMAGE_TOPIC,
                                          Image,
                                          self.color_callback, queue_size=1, buff_size=10000000)
        self.depth_sub = rospy.Subscriber(DEPTH_MAP_TOPIC,
                                          Image,
                                          self.depth_callback, queue_size=1, buff_size=10000000)
        self.camera_sub = rospy.Subscriber('CameraPose_ros',PoseStamped, self.camera_callback, queue_size=1, buff_size=10000000)

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

    def color_callback(self, data):
        """ Called whenever color data is available. """
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.color_mutex.acquire()
        self.color_image = cv_image
        self.color_mutex.release()

    def depth_callback(self, data):
        """ Called whenever depth data is available. """
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        self.depth_mutex.acquire()
        self.depth_image = cv_image
        self.depth_mutex.release()

    def camera_callback(self, data):
        """ Called whenever depth data is available. """
        self.camera_mutex.acquire()
        self.position = [data.pose.position.x, data.pose.position.z, data.pose.position.y]
        self.rotation = [data.pose.orientation.x, data.pose.orientation.z, data.pose.orientation.y, data.pose.orientation.w]
        self.camera_mutex.release()

    def cal_quaternion(self, gaze_center):
        theta1 = np.arctan2(gaze_center[2], gaze_center[0])
        theta2 = np.arctan2(gaze_center[1], gaze_center[0]) 
        return theta1, theta2

    def person2marker(self, pub, br, camera_frame_id, coord3d_mat, vis_mat):
        shift = 0

        KEYPOINT_NAME_DICT = {0: "Nose", 1: "Neck",
                                2: "RShoulder", 3: "RElbow", 4: "RWrist",
                                5: "LShoulder", 6: "LElbow", 7: "LWrist",
                                8: "RWaist", 9: "RKnee", 10: "RAnkle",
                                11: "LWaist", 12: "LKnee", 13: "LAnkle",
                                14: "REye", 15: "LEye", 16: "REar", 17: "LEar"}
        if self.center is not None and self.thetas is not None:
            br.sendTransform(self.cam_fixed_frame[0],
                                self.cam_fixed_frame[1],
                                rospy.Time.now(),
                                'camera_fixed_frame',
                                'map')
            # euler = transformations.euler_from_quaternion([self.rotation[0], -self.rotation[1], self.rotation[2], self.rotation[3]])
            q1, q3, q2, q0 = self.rotation[0], -self.rotation[1], self.rotation[2], self.rotation[3]
            euler = [0, 0, 0]
            euler[1] = math.atan2(2*(q0*q3 + q1*q2), 1- 2*(q2**2 + q3**2))
            br.sendTransform((self.center[0] + self.position[0], self.center[1] - shift - self.position[1], self.center[2] + self.position[2]),
                                    transformations.quaternion_from_euler(0, -self.thetas[0] + euler[1], 0),
                                    rospy.Time.now(),
                                    'human_frame',
                                    'camera_fixed_frame')

            br.sendTransform((0, 0, 0),
                                    transformations.quaternion_from_euler(0, -1.57, 0),
                                    rospy.Time.now(),
                                    'marker_frame',
                                    'human_frame')

            ma = MarkerArray()
            h = Header(frame_id='marker_frame')
            ma.markers.append(self.line_list)
            pub.publish(ma)
            br.sendTransform((self.center[0], self.center[1] - shift, self.center[2]),
                                    transformations.quaternion_from_euler(0, -self.thetas[0], 0),
                                    rospy.Time.now(),
                                    'ar_frame',
                                    'camera_fixed_frame')
            print('send')
            try:
                (trans_m, rot_m) = self.tflistener.lookupTransform('map', 'marker_frame', rospy.Time(0))
            except (LookupException, ConnectivityException, ExtrapolationException):
                return

            """ Publishes detected persons as ROS line lists. """
        else:
            LIMBS = np.array([[1, 2], [2, 3], [3, 4],
                        [1, 8], [8, 9], [9, 10],
                        [1, 5], [5, 6], [6, 7],
                        [1, 11], [11, 12], [12, 13],
                        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]])


            ma = MarkerArray()
            h = Header(frame_id='marker_frame')
            line_list = Marker(type=Marker.LINE_LIST, id=0)
            line_list.header = h
            line_list.action = Marker.ADD
            line_list.scale.x = 0.03
            points_to_calavg = []
            id_to_calavg = [0, 1, 2, 5, 14, 14, 16, 17]
            self.coor_mat = coord3d_mat
            for pid in range(coord3d_mat.shape[0]):
                # broadcast keypoints as tf
                for kid in range(coord3d_mat.shape[1]):
                    if kid in id_to_calavg and vis_mat[pid, kid]:
                        points_to_calavg.append(coord3d_mat[pid, kid, :])
                    br.sendTransform((coord3d_mat[pid, kid, 0], coord3d_mat[pid, kid, 1]-shift, coord3d_mat[pid, kid, 2]),
                        transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "/human_pose/person%d/%s" % (pid, KEYPOINT_NAME_DICT[kid]),
                        camera_frame_id)

                if vis_mat[pid, 14] and vis_mat[pid, 15] and vis_mat[pid, 0]:
                    x_l, y_l, z_l = coord3d_mat[pid, 14]
                    x_r, y_r, z_r = coord3d_mat[pid, 15]
                    x_n, y_n, z_n = coord3d_mat[pid, 0]
                    center = [np.mean([x_l, x_r]), np.mean([y_l, y_r]), np.mean([z_l, z_r])]
                else:
                    return

                # draw skeleton figure
                for lid, (p0, p1) in enumerate(LIMBS):
                    if vis_mat[pid, p0] and vis_mat[pid, p1]:
                        self.points.append([coord3d_mat[pid, p0, 0] - center[0], coord3d_mat[pid, p0, 1]-shift - center[1], coord3d_mat[pid, p0, 2] - center[2]])
                        self.points.append([coord3d_mat[pid, p1, 0] - center[0], coord3d_mat[pid, p1, 1]-shift - center[1], coord3d_mat[pid, p1, 2] - center[2]])
                        p0 = Point(x=coord3d_mat[pid, p0, 0] - center[0],
                                y=coord3d_mat[pid, p0, 1]-shift - center[1],
                                z=coord3d_mat[pid, p0, 2] - center[2])
                        p1 = Point(x=coord3d_mat[pid, p1, 0] - center[0],
                                y=coord3d_mat[pid, p1, 1]-shift - center[1],
                                z=coord3d_mat[pid, p1, 2] - center[2])
                        line_list.points.append(p0)
                        line_list.points.append(p1)
                        

                if vis_mat[pid, 14] and vis_mat[pid, 15] and vis_mat[pid, 0]:
                    x_l, y_l, z_l = coord3d_mat[pid, 14]
                    x_r, y_r, z_r = coord3d_mat[pid, 15]
                    x_n, y_n, z_n = coord3d_mat[pid, 0]
                    center = [np.mean([x_l, x_r]), np.mean([y_l, y_r]), np.mean([z_l, z_r])]
                    normal = np.array([x_n - center[0], y_n - center[1], z_n - center[2]]) #+ np.array([0, 0, 0.05])
                    if np.linalg.norm(normal) > 0:
                        normal = normal/np.linalg.norm(normal)

                    theta1, theta2 = self.cal_quaternion(normal)
                    if self.center is None or self.thetas is None:
                        try:
                            (trans_cam, rot_cam) = self.tflistener.lookupTransform('map', camera_frame_id, rospy.Time(0))
                        except (LookupException, ConnectivityException, ExtrapolationException):
                            return
                        #pub a fixed camera link
                        self.center = center
                        self.thetas = [theta1, theta2]
                        self.cam_fixed_frame = [trans_cam, rot_cam]

                        br.sendTransform(self.cam_fixed_frame[0],
                                            self.cam_fixed_frame[1],
                                            rospy.Time.now(),
                                            'camera_fixed_frame',
                                            'map')

                        br.sendTransform((center[0], center[1] - shift, center[2]),
                                            transformations.quaternion_from_euler(0, -theta1, 0),
                                            rospy.Time.now(),
                                            'human_frame',
                                            'camera_fixed_frame')
                        br.sendTransform((0, 0, 0),
                                            transformations.quaternion_from_euler(0, -1.57, 0),
                                            rospy.Time.now(),
                                            'ar_frame',
                                            'human_frame')
                        br.sendTransform((0, 0, 0),
                                            transformations.quaternion_from_euler(0, 0, 0),
                                            rospy.Time.now(),
                                            'marker_frame',
                                            'camera_fixed_frame')


                line_list.color.r = 1.0
                line_list.color.g = 0.0
                line_list.color.b = 0.0
                line_list.color.a = 1.0
                self.line_list = line_list
                ma.markers.append(line_list)
                pub.publish(ma)


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
    # read calib from ros topic
    camera_calib = CameraCalibSubscriber(CAMERA_INFO_TOPIC)
    rospy.init_node('listener', anonymous=True)
    K, camera_frame_id = camera_calib.wait_for_calib()

    # Start Node that reads the kinect topic
    data = KinectDataSubscriber()
    rospy.init_node('listener', anonymous=True) 

    # create pose network
    poseNet = PoseNet3D(ope_depth=OPE_DEPTH, vpn_type=VPN_TYPE,
                        gpu_id=GPU_ID, gpu_memory_limit=GPU_MEMORY,
                        K=K)

    pub = rospy.Publisher('human_pose', MarkerArray, queue_size=1)
    br = TransformBroadcaster()

    # loop
    try:
        while not rospy.is_shutdown():
            data.color_mutex.acquire()
            data.depth_mutex.acquire()
            data.camera_mutex.acquire()
            if (data.color_image is not None) and (data.depth_image is not None) and (data.position is not None):
                color = data.color_image.copy()
                depth = data.depth_image.copy()
                mask = np.logical_not(depth == 0.0)
                data.color_mutex.release()
                data.depth_mutex.release()
                data.camera_mutex.release()
                # run algorithm
                coords_pred, det_conf = poseNet.detect(color, depth, mask)

                # publish results
                data.person2marker(pub, br, camera_frame_id, coords_pred, det_conf > CONF_THRESH)
            else:
                data.color_mutex.release()
                data.depth_mutex.release()
                data.camera_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
