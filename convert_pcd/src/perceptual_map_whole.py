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
from threading import Thread, Lock
import math
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import time

class PerceptualMap:
    def __init__(self):
        rospy.init_node('voxel_listener', anonymous=True)
        self.tflistener = tf.TransformListener()
        self.table_sub = rospy.Subscriber('table_world', PointCloud2, self.table_callback, queue_size=1)
        self.sphere1_sub = rospy.Subscriber('sphere1_world', PointCloud2, self.sphere1_callback, queue_size=1)
        self.bottle_sub = rospy.Subscriber('bottle_world', PointCloud2, self.bottle_callback, queue_size=1)
        self.meat_sub = rospy.Subscriber('meat_world', PointCloud2, self.meat_callback, queue_size=1)
        self.potato_sub = rospy.Subscriber('potato_world', PointCloud2, self.potato_callback, queue_size=1)
        self.tomato_sub = rospy.Subscriber('tomato_world', PointCloud2, self.tomato_callback, queue_size=1)
        self.tomato2_sub = rospy.Subscriber('tomato2_world', PointCloud2, self.tomato2_callback, queue_size=1)
        self.veggie_sub = rospy.Subscriber('veggie_world', PointCloud2, self.veggie_callback, queue_size=1)
        self.map_sub = rospy.Subscriber('map_cloud_pcd', PointCloud2, self.map_callback, queue_size=1)
        self.br = tf.TransformBroadcaster()
        self.pub_test = rospy.Publisher('visual_map', PointCloud2, queue_size=1)
        self.pub_test_temp = rospy.Publisher('test_table', PointCloud2, queue_size=1)
        self.pub_ar_flags = rospy.Publisher('ar_flags', numpy_msg(Floats), queue_size=1)
        self.pub_id_flags = rospy.Publisher('id_flags', numpy_msg(Floats), queue_size=1)

        self.table_mutex = Lock()
        self.sphere1_mutex = Lock()
        self.bottle_mutex = Lock()
        self.meat_mutex = Lock()
        self.tomato_mutex = Lock()
        self.tomato2_mutex = Lock()
        self.potato_mutex = Lock()
        self.veggie_mutex = Lock()
        self.map_mutex = Lock()

        self.table_points = None
        self.sphere1_points = None
        self.bottle_points = None
        self.meat_points = None
        self.potato_points = None
        self.tomato_points = None
        self.tomato2_points = None
        self.veggie_points = None
        self.map_points = None

        self.table_colors = None
        self.sphere1_colors = None
        self.bottle_colors = None
        self.meat_colors = None
        self.potato_colors = None
        self.tomato_colors = None
        self.tomato2_colors = None
        self.veggie_colors = None
        self.map_colors = None

        self.total_maps = None
        self.total_colors = None

        self.map_flags = None
        self.display_points = None
        self.display_colors = None
        self.ar_flags = None
        self.id_flags = None
        self.frame_trans = None
        self.frame_rot = None
        self.obj_num = 7#obj without map

    def sphere1_callback(self, data):
        self.sphere1_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            points.append(point)
            colors.append([50.0/255,205.0/255,50.0/255])
        self.sphere1_mutex.release()
        self.sphere1_points = np.array(points)
        self.sphere1_colors = np.array(colors)

    def table_callback(self, data):
        self.table_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.table_mutex.release()
        self.table_points = np.array(points)
        self.table_colors = np.array(colors)

    def bottle_callback(self, data):
        self.bottle_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.bottle_mutex.release()
        self.bottle_points = np.array(points)
        self.bottle_colors = np.array(colors)

    def meat_callback(self, data):
        self.meat_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.meat_mutex.release()
        self.meat_points = np.array(points)
        self.meat_colors = np.array(colors)

    def potato_callback(self, data):
        self.potato_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.potato_mutex.release()
        self.potato_points = np.array(points)
        self.potato_colors = np.array(colors)

    def tomato_callback(self, data):
        self.tomato_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.tomato_mutex.release()
        self.tomato_points = np.array(points)
        self.tomato_colors = np.array(colors)

    def tomato2_callback(self, data):
        self.tomato2_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.tomato2_mutex.release()
        self.tomato2_points = np.array(points)
        self.tomato2_colors = np.array(colors)

    def veggie_callback(self, data):
        self.veggie_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            colors.append([p[3],p[4],p[5]])
        self.veggie_mutex.release()
        self.veggie_points = np.array(points)
        self.veggie_colors = np.array(colors)

    def map_callback(self, data):
        self.map_mutex.acquire()
        points = []
        colors = []
        for p in pc2.read_points(data, field_names = ("x", "y", "z", "rgb"), skip_nans=True):
            point = np.array([p[0],p[1],p[2]])
            point = np.array(point)
            points.append(point)
            
            test = p[3] 
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            colors.append([r/255.0,g/255.0,b/255.0])
            # colors.append([p[3], p[4], p[5]])
        self.map_mutex.release()
        self.map_points = np.array(points)
        self.map_colors = np.array(colors)

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

    
    def check_visual_level(self, rot_angle, axis, map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level, flag=False):
        plane_in_h = plane_in_h - human_center
        map_points = map_points - human_center
        trans_ar_in_h = trans_ar_in_h - human_center

        theta1, theta2 = self.cal_quaternion(trans_ar_in_h)
        rot = R.from_euler('z', -theta1)
        map_rot = rot.apply(map_points) 
        plane_in_h = rot.apply(plane_in_h)

        rot = R.from_euler(axis, rot_angle, degrees=True)
        map_rot = rot.apply(map_rot)
        y_min = np.min(plane_in_h[:, 1])
        y_max = np.max(plane_in_h[:, 1])
        z_min = np.min(plane_in_h[:, 2])
        z_max = np.max(plane_in_h[:, 2])

        candidate_points_id = np.logical_and(np.logical_and(np.logical_and(np.logical_and(map_rot[:, 1] <= y_max, map_rot[:, 1]>=y_min), 
                                map_rot[:, 2]<=z_max), map_rot[:, 2]>=z_min), map_rot[:, 0] >= 0.3)
        candidate_points_ids = np.where(candidate_points_id)
        

        level2_1_dict = dict()
        if len(candidate_points_ids[0]) > 0:
            map_rot = np.round(map_rot, decimals = 2)
            plane_in_h = np.round(plane_in_h, decimals = 2)
            
            for pid in candidate_points_ids[0]:
                x, y, z = map_rot[pid]
                y_idx = (y - y_min)/0.05
                z_idx = (z - z_min)/0.05
                indx = int(z_idx*100 + y_idx)
                if indx in level2_1_dict:
                    if np.mean(map_points[level2_1_dict[indx]][:, 0]) > x:
                        level2_1_dict[indx] = [pid]
                else:
                    level2_1_dict[indx] = [pid]

        points = np.empty((0, 3))
        colors = np.empty((0, 3))
        temp_flags = np.zeros(map_points.shape[0]).astype(int)
        flags = np.empty((0, 1))
        for key in level2_1_dict.keys():
            temp_flags[level2_1_dict[key]] = 1
            temp_flags = np.multiply(temp_flags, map_flags).astype(bool)

            points = np.vstack([points, map_points[temp_flags]])
            colors = np.vstack([colors, map_colors[temp_flags]])
            map_flags[temp_flags] = 0
            if id_flags is not None:
                flags = np.vstack([flags, id_flags[temp_flags].reshape(-1, 1)])

        return points, colors, map_flags, flags

    def rot_ar(self, sphere, frame_id):
        try:
            (trans, rot) = self.tflistener.lookupTransform('map', frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None

        r_ar = R.from_quat(rot)
        sphere_points = sphere
        ar_points = r_ar.apply(sphere_points) + np.array(trans)

        return ar_points, np.array(trans)


    def check_visual(self, map_points, map_colors, id_flags):
        print('map_visual_check')

        self.br.sendTransform((5, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'map_in_human',
                            'human_frame')

        z_min, z_max, y_min, y_max = -1, 1, -1.5, 1
        step = 0.02
        coordinates = list(product([0], np.arange(y_min, y_max, step), np.arange(z_min, z_max, step))) 
        stdplane = np.array(coordinates)

        try:
            (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

        human_center = np.array(trans_h)

        try:
            (trans_ar_in_h, rot_ar_in_h) = self.tflistener.lookupTransform('map', 'map_in_human', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

        total_points = np.empty((0, 3))
        total_colors = np.empty((0, 3))
        total_flags = np.empty((0, 1))
        self.ar_flags = np.zeros(map_points.shape[0])
        #level1
        print('level1....')
        col_level = 2
        start = time.time()
        r_ar = R.from_quat(rot_ar_in_h)
        plane_in_h = r_ar.apply(stdplane) + np.array(trans_ar_in_h)
        map_flags = np.ones(map_points.shape[0])
        ground_angle = 5
        rot_angle = 20

        # for angle in range(0, 2*rot_angle + 1, rot_angle):
        #     points, colors, map_flags, flags = self.check_visual_level([rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #     colors[:, col_level] = colors[:, col_level]*0.25
        #     total_points = np.vstack([total_points, points])
        #     total_colors = np.vstack([total_colors, colors])   
        #     total_flags = np.vstack([total_flags, flags]) 

        #     if angle != -angle:
        #         points, colors, map_flags, flags = self.check_visual_level([-rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #         colors[:, col_level] = colors[:, col_level]*0.25
        #         total_points = np.vstack([total_points, points])
        #         total_colors = np.vstack([total_colors, colors])   
        #         total_flags = np.vstack([total_flags, flags]) 

        #     points, colors, map_flags, flags = self.check_visual_level([rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #     colors[:, col_level] = colors[:, col_level]*0.25
        #     total_points = np.vstack([total_points, points])
        #     total_colors = np.vstack([total_colors, colors])   
        #     total_flags = np.vstack([total_flags, flags]) 

        #     if angle != -angle:
        #         points, colors, map_flags, flags = self.check_visual_level([-rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #         colors[:, col_level] = colors[:, col_level]*0.25
        #         total_points = np.vstack([total_points, points])
        #         total_colors = np.vstack([total_colors, colors])   
        #         total_flags = np.vstack([total_flags, flags]) 


        points, colors, map_flags, flags = self.check_visual_level([0], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])   
        total_flags = np.vstack([total_flags, flags]) 
        

        points, colors, map_flags, flags = self.check_visual_level([-rot_angle], 'y', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        temp_shape = total_points.shape[0]
        end = time.time()

        points, colors, map_flags, flags = self.check_visual_level([rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        temp_shape = total_points.shape[0]


        points, colors, map_flags, flags = self.check_visual_level([2*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-2*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([2*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-2*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        
        self.ar_flags[:total_points.shape[0]] = 1
        temp_shape = total_points.shape[0]

        #level2
        print('level2...')
        col_level = 0
        # for angle in range(3*rot_angle, 180 + 1, rot_angle):
        #     points, colors, map_flags, flags = self.check_visual_level([rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #     colors[:, col_level] = colors[:, col_level]*0.25
        #     total_points = np.vstack([total_points, points])
        #     total_colors = np.vstack([total_colors, colors])   
        #     total_flags = np.vstack([total_flags, flags]) 

        #     if angle != -angle:
        #         points, colors, map_flags, flags = self.check_visual_level([-rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #         colors[:, col_level] = colors[:, col_level]*0.25
        #         total_points = np.vstack([total_points, points])
        #         total_colors = np.vstack([total_colors, colors])   
        #         total_flags = np.vstack([total_flags, flags]) 

        #     points, colors, map_flags, flags = self.check_visual_level([rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #     colors[:, col_level] = colors[:, col_level]*0.25
        #     total_points = np.vstack([total_points, points])
        #     total_colors = np.vstack([total_colors, colors])   
        #     total_flags = np.vstack([total_flags, flags]) 

        #     if angle != -angle:
        #         points, colors, map_flags, flags = self.check_visual_level([-rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        #         colors[:, col_level] = colors[:, col_level]*0.25
        #         total_points = np.vstack([total_points, points])
        #         total_colors = np.vstack([total_colors, colors])   
        #         total_flags = np.vstack([total_flags, flags]) 
        points, colors, map_flags, flags = self.check_visual_level([4*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-4*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([4*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-4*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        self.ar_flags[temp_shape:total_points.shape[0]] = 2
        temp_shape = total_points.shape[0]

        points, colors, map_flags, flags = self.check_visual_level([3*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-3*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([3*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-3*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([5*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-5*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([5*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-5*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, col_level] = colors[:, col_level]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([6*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-6*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([6*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-6*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([7*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-7*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([7*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-7*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([8*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-8*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([8*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([-8*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([9*rot_angle], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 

        # points, colors, map_flags, flags = self.check_visual_level([9*rot_angle, -ground_angle], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        # colors[:, col_level] = colors[:, col_level]*0.25
        # total_points = np.vstack([total_points, points])
        # total_colors = np.vstack([total_colors, colors])
        # total_flags = np.vstack([total_flags, flags]) 


        self.ar_flags[temp_shape:total_points.shape[0]] = 2
        temp_shape = total_points.shape[0]
        print('level:{}'.format(time.time()-start))
        #level3
        map_flags = map_flags.astype(bool)
        colors = map_colors[map_flags]
        colors = colors*0.75
        total_points = np.vstack([total_points, map_points[map_flags] - human_center])
        total_colors = np.vstack([total_colors, colors])
        if id_flags is not None:
            flags = id_flags[map_flags].reshape((-1, 1))
            total_flags = np.vstack([total_flags, flags]) 
            self.pub_id_flags.publish(total_flags.astype(np.float32))
            self.id_flags = total_flags
        self.ar_flags[temp_shape:] = 3

        self.display_points = total_points + human_center
        self.display_colors = total_colors
        msg = self.xyzrgb_array_to_pointcloud2(total_points + human_center, total_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
        self.pub_test.publish(msg)
        self.br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'visual_map',
                            'map')

        self.pub_ar_flags.publish(self.ar_flags.astype(np.float32))
        print('sent')


    def pointcloud2withrgb(self, points, colors, stamp, frame_id, seq):
        points_array = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            color = colors[i]*255
            color = color.astype(int)
            hex_r = (0xff & color[0]) << 16
            hex_g = (0xff & color[1]) << 8
            hex_b = (0xff & color[2])

            hex_rgb = hex_r | hex_g | hex_b

            float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]
            # r, g, b = color.astype(int)
            # a = 0
            # rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, float_rgb]
            points_array.append(pt)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.FLOAT32, 1),
                ]

        header = Header()
        header.frame_id = frame_id
        msg = pc2.create_cloud(header, fields, points_array)
        msg.header.stamp = stamp
        return msg


    def callback(self):
        print('points recevied')
        
        if self.display_colors is not None:
            print('publish existed')
            msg = self.xyzrgb_array_to_pointcloud2(self.display_points, self.display_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
            self.pub_test.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                'visual_map',
                                'map')
            self.pub_ar_flags.publish(self.ar_flags.astype(np.float32))
            self.pub_id_flags.publish(self.id_flags.astype(np.float32))
            return
        
        map_points = self.map_points
        map_colors = self.map_colors
        flags = np.zeros(self.obj_num)
        self.id_flags = np.zeros(map_points.shape[0])

        check_tables = [self.table_points, self.bottle_points, self.meat_points, self.potato_points, self.tomato_points, self.tomato2_points, self.veggie_points]
        check_colors = [self.table_colors, self.bottle_colors, self.meat_colors, self.potato_colors, self.tomato_colors, self.tomato2_colors, self.veggie_colors]
        check_frame_ids = ['table_points', 'bottle_points', 'meat_points', 'potato_points', 'tomato_points', 'tomato2_points', 'veggie_points']

        try:
            (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

        frame_rot = np.array(rot_h)
        frame_trans = np.array([trans_h])
        for obj_id, check_obj in enumerate(check_tables):
            table_points, trans = self.rot_ar(check_obj, check_frame_ids[obj_id])
            if trans is None:
                return
            frame_trans = np.vstack([frame_trans, trans]) 
            map_points = np.vstack([map_points, table_points])
            map_colors = np.vstack([map_colors, check_colors[obj_id]])
            flags[obj_id] = 1
            self.id_flags = np.hstack([self.id_flags, (obj_id + 1)*np.ones(table_points.shape[0])])
        
        if self.display_points is None:        
            self.frame_trans = frame_trans
            self.frame_rot = frame_rot
            self.ar_flags = np.zeros(map_points.shape[0])
            self.check_visual(map_points, map_colors, self.id_flags)
        else:
            dist_trans = np.linalg.norm(self.frame_trans - frame_trans)
            dist_rot = np.linalg.norm(self.frame_rot - frame_rot)
            if dist_trans > 0.5 or dist_rot > 0.5 or self.ar_flags.shape[0] != map_points.shape[0]:
                self.ar_flags = np.zeros(map_points.shape[0])
                self.check_visual(map_points, map_colors, self.id_flags)
            else:
                msg = self.pointcloud2withrgb(self.display_points, self.display_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
                self.pub_test.publish(msg)
                self.br.sendTransform((0, 0, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, 0),
                                    rospy.Time.now(),
                                    'visual_map',
                                    'map')
                self.pub_ar_flags.publish(self.ar_flags.astype(np.float32))
                self.pub_id_flags.publish(self.id_flags.astype(np.float32))
                print('exist published')
                
                

if __name__ == '__main__':
    mapper = PerceptualMap()
    try:
        while not rospy.is_shutdown():
            mapper.table_mutex.acquire()
            mapper.map_mutex.acquire()
            mapper.sphere1_mutex.acquire()
            mapper.bottle_mutex.acquire()
            mapper.meat_mutex.acquire()
            mapper.potato_mutex.acquire()
            mapper.tomato_mutex.acquire()
            mapper.tomato2_mutex.acquire()
            mapper.veggie_mutex.acquire()
            if (mapper.table_points is not None) \
                and (mapper.map_points is not None) and (mapper.sphere1_points is not None) and (mapper.bottle_points is not None) and \
                (mapper.meat_points is not None) and (mapper.potato_points is not None) and \
                (mapper.tomato_points is not None) and (mapper.tomato2_points is not None) and (mapper.veggie_points is not None):
                mapper.callback()
                mapper.table_mutex.release()
                mapper.map_mutex.release()
                mapper.sphere1_mutex.release()
                mapper.bottle_mutex.release()
                mapper.meat_mutex.release()
                mapper.potato_mutex.release()
                mapper.tomato_mutex.release()
                mapper.tomato2_mutex.release()
                mapper.veggie_mutex.release()
            else:
                mapper.table_mutex.release()
                mapper.map_mutex.release()
                mapper.sphere1_mutex.release()
                mapper.bottle_mutex.release()
                mapper.meat_mutex.release()
                mapper.potato_mutex.release()
                mapper.tomato_mutex.release()
                mapper.tomato2_mutex.release()
                mapper.veggie_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
    

