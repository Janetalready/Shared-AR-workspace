#!/usr/bin/env python
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point, PointStamped
import numpy as np
#from openpose_ros_msgs.msg import PersonDetection_3d
import tf
from std_msgs.msg import String, Header
from itertools import product
from sympy import Point3D, Plane
from sympy.geometry import Line3D
import sympy
from transformations import rotation_matrix, angle_between_vectors, vector_product
import ctypes
import struct
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

class SlamAR:
    def __init__(self):
        rospy.init_node('voxel_listener', anonymous=True)
        self.tflistener = tf.TransformListener()
        self.pub_now_virtual = dict()
        self.pub_total_virtual = dict()
        self.pub_total_virtual['cube_points'] = rospy.Publisher('sphere_total_virtual', PointCloud2, queue_size=1)
        self.pub_total_virtual['sphere2_points'] = rospy.Publisher('sphere2_total_virtual', PointCloud2, queue_size=1)
        self.pub_total_virtual['cylinder_points'] = rospy.Publisher('cylinder_total_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['cube_points'] = rospy.Publisher('sphere_now_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['sphere2_points'] = rospy.Publisher('sphere2_now_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['cylinder_points'] = rospy.Publisher('cylinder_now_virtual', PointCloud2, queue_size=1)
        self.pub_test = rospy.Publisher('sphere_test', PointCloud2, queue_size=1)
        # pub = rospy.Publisher('voxel_update', PointCloud2, queue_size=5)
        self.br = tf.TransformBroadcaster()
        self.virtual_points = dict()
        self.new_points = None
        self.flag = {'cube_points':1, 'sphere2_points':1, 'cylinder_points':1}

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        msg = PointCloud2()
        assert(points.shape == colors.shape)

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq: 
            msg.header.seq = seq
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            N = len(points)
            # print(N)
            xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            msg.height = 1
            msg.width = N
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * N
        msg.is_dense = True; 
        msg.data = xyzrgb.tostring()
        return msg 


    def cal_quaternion(self, gaze_center):
        theta1 = np.arctan2(gaze_center[1], gaze_center[0])
        theta2 = np.arctan2(gaze_center[2], gaze_center[0])
        return theta1, theta2

    def plane(self, a, b, c):
        p1 = b - a
        p2 = c - a
        normal = np.cross(p2, p1)
        D = -normal.dot(a)

        return normal[0], normal[1], normal[2], D

    def check_in_plane(self, planes, point):
        flag = 1
        for plane in planes:
            if plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3] < 0:
                flag = 0
                break
        return flag

    def cos_angle(self, slam_point, std_point, curr_position):
        p1 = slam_point - curr_position
        p2 = std_point - curr_position
        p1 = p1/np.linalg.norm(p1)
        p2 = p2/np.linalg.norm(p2)
        return p1.dot(p2)

    def check_ar(self, frame_id, sphere, slam):
        try:
            (trans, rot) = self.tflistener.lookupTransform('map', frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

        try:
            (trans_cam, rot_cam) = self.tflistener.lookupTransform('map', 'camera_mount_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return  

        # try:
        #     (trans_s, rot_s) = self.tflistener.lookupTransform('ar_frame', frame_id, rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return 
        print(frame_id)
        # print(rot_s)
        sphere_points = np.array(list(pc2.read_points(sphere, field_names = ("x", "y", "z"), skip_nans=True)))

        r_ar = R.from_quat(rot)
        sphere_points_ar = r_ar.apply(sphere_points) + np.array(trans)
        
        z_min = min(sphere_points[:, 2])
        z_max = max(sphere_points[:, 2])
        y_min = min(sphere_points[:, 1])
        y_max = max(sphere_points[:, 1])
        step = 0.02
        coordinates = list(product([0], np.arange(y_min, y_max, step), np.arange(z_min, z_max, step))) + [[0, y_min, z_min], [0, y_max, z_min], [0, y_min, z_max], [0, y_max, z_max]]
        stdplane = np.array(coordinates)

        inverse_normal = np.array([-trans[0], -trans[1], -trans[2]])
        x_axis = np.array([-1, 0, 0])
        rotate_theta = np.dot(inverse_normal/np.linalg.norm(inverse_normal), x_axis)
        rotate_angle = np.arccos(rotate_theta)

        r = R.from_euler('z', rotate_angle)
        pre_rotate_stdplane = r.apply(stdplane)
        plane_ar = r_ar.apply(pre_rotate_stdplane) + np.array(trans)
        curr_position = np.array(trans_cam)

        # plane1 = self.plane(stdplane[-2], stdplane[-1], stdplane[-4])
        # plane2 = self.plane(curr_position, stdplane[-2], stdplane[-4])
        # plane3 = self.plane(curr_position, stdplane[-3], stdplane[-1])
        # plane4 = self.plane(curr_position, stdplane[-1], stdplane[-2])
        # plane5 = self.plane(curr_position, stdplane[-4], stdplane[-3])
        # planes = [plane1, plane2, plane3, plane4, plane5]

        show_points = []
        prj_dis = dict()
        slam_points = np.array(list(pc2.read_points(slam, field_names = ("x", "y", "z"), skip_nans=True)))
        for pid, std_point in enumerate(plane_ar[:-4]):
            flag = 0
            for slam_point in slam_points:
                dist = np.linalg.norm(slam_point - curr_position)
                if self.cos_angle(slam_point, std_point, curr_position) > 0.995 and dist < np.linalg.norm(std_point - curr_position):
                    prj_dis[pid] = dist
                    flag = 1
            if flag == 0:
                prj_dis[pid] = 1000
        
        for std_point in plane_ar[:-4]:
            for virtual_point in sphere_points_ar:
                dist = np.linalg.norm(virtual_point - curr_position)
                if self.cos_angle(slam_point, std_point, curr_position) > 0.995 and dist < prj_dis[pid]:
                    show_points.append(virtual_point)
        
        
        colors = np.ones(plane_ar.shape)
        if frame_id == 'cube_points':
            colors[:, 0] = 255.0/255
            colors[:, 1] = 192.0/255
            colors[:, 2] = 203.0/255
        elif frame_id == 'sphere2_points':
            colors[:, 0] = 1
            colors[:, 1] = 1
            colors[:, 2] = 1
        else:
            colors[:, 0] = 1
            colors[:, 1] = 1
            colors[:, 2] = 0

        if frame_id == "cylinder_points":
            msg = self.xyzrgb_array_to_pointcloud2(plane_ar, colors, stamp=rospy.get_rostime(), frame_id=frame_id + '_test', seq=None)
            self.pub_test.publish(msg)
            # self.br.sendTransform((trans[0], trans[1], trans[2]),
            #                     (rot[0], rot[1], rot[2], rot[3]),
            #                     rospy.Time.now(),
            #                     frame_id + '_test',
            #                     'map')
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                frame_id + '_test',
                                'map')




    def callback(self, sphere2, cylinder, sphere, slam):
        print('points recevied')
        self.check_ar('cube_points', sphere, slam)
        self.check_ar('sphere2_points', sphere2, slam)
        self.check_ar('cylinder_points', cylinder, slam)


    def listener(self):
        sphere2_points_sub = message_filters.Subscriber('sphere2_world', PointCloud2)
        cylinder_points_sub = message_filters.Subscriber('cylinder_world', PointCloud2)
        sphere_points_sub = message_filters.Subscriber('cube_world', PointCloud2)
        slam_points_sub = message_filters.Subscriber('map_update', PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([sphere2_points_sub, cylinder_points_sub, sphere_points_sub, slam_points_sub], 10, 1, allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.spin()

if __name__ == '__main__':
    slam_ar = SlamAR()
    slam_ar.listener()
    

