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

class PerceptualMap:
    def __init__(self):
        rospy.init_node('voxel_listener', anonymous=True)
        self.tflistener = tf.TransformListener()
        self.buuny_sub = rospy.Subscriber('bunny_world', PointCloud2, self.bunny_callback, queue_size=1)
        self.teapot_sub = rospy.Subscriber('teapot_world', PointCloud2, self.teapot_callback, queue_size=1)
        self.table_sub = rospy.Subscriber('table_world', PointCloud2, self.table_callback, queue_size=1)
        self.map_sub = rospy.Subscriber('map_update', PointCloud2, self.map_callback, queue_size=1)
        self.br = tf.TransformBroadcaster()
        self.pub_test = rospy.Publisher('visual_map', PointCloud2, queue_size=1)
        self.pub_test_temp = rospy.Publisher('test_table', PointCloud2, queue_size=1)
        self.pub_ar_flags = rospy.Publisher('ar_flags', numpy_msg(Floats), queue_size=1)
        self.pub_id_flags = rospy.Publisher('id_flags', numpy_msg(Floats), queue_size=1)

        self.bunny_mutex = Lock()
        self.teapot_mutex = Lock()
        self.table_mutex = Lock()
        self.map_mutex = Lock()

        self.bunny_points = None
        self.teapot_points = None
        self.table_points = None
        self.map_points = None
        self.bunny_colors = None
        self.teapot_colors = None
        self.table_colors = None
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


    def bunny_callback(self, data):
        self.bunny_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True)):
            point = np.array([p[0],p[1],p[2]])
            points.append(point)
            colors.append([0, 172.0/255, 223.0/255])
        self.bunny_mutex.release()
        self.bunny_points = np.array(points)
        self.bunny_colors = np.array(colors)

    def teapot_callback(self, data):
        self.teapot_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
                point = np.array([p[0],p[1],p[2]])
                point = np.array(point)
                points.append(point)
                colors.append([p[3],p[4],p[5]])
        self.teapot_mutex.release()
        self.teapot_points = np.array(points)
        self.teapot_colors = np.array(colors)

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

    def map_callback(self, data):
        self.map_mutex.acquire()
        points = []
        colors = []
        for p in pc2.read_points(data, field_names = ("x", "y", "z","rgb"), skip_nans=True):
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
        
        # if flag:
        #     msg = self.xyzrgb_array_to_pointcloud2(candidate_points, map_colors[candidate_points_id], stamp=rospy.get_rostime(), frame_id='test', seq=None)
        #     self.pub_test_temp.publish(msg)
        #     self.br.sendTransform((0, 0, 0),
        #                         tf.transformations.quaternion_from_euler(0, 0, 0),
        #                         rospy.Time.now(),
        #                         'test',
        #                         'map')

        level2_1_dict = dict()
        if len(candidate_points_ids[0]) > 0:
            map_rot = np.round(map_rot, decimals = 2)
            plane_in_h = np.round(plane_in_h, decimals = 2)
            
            for pid in candidate_points_ids[0]:
                x, y, z = map_rot[pid]
                y_idx = (y + 1)/0.02
                z_idx = (z + 1)/0.02
                indx = int(z_idx*100 + y_idx)
                if indx in level2_1_dict:
                    if np.mean(map_points[level2_1_dict[indx]][:, 0]) > x:
                        if abs(np.mean(map_points[level2_1_dict[indx]][:, 0]) - x) <= 0.02:
                            level2_1_dict[indx].append(pid)
                        else:
                            level2_1_dict[indx] = [pid]
                else:
                    level2_1_dict[indx] = [pid]

        points = np.empty((0, 3))
        colors = np.empty((0, 3))
        temp_flags = np.zeros(map_points.shape[0])
        flags = np.empty((0, 1))
        for key in level2_1_dict.keys():
            # joint_ids = set(np.where(self.map_flags == 0)[0]).intersection(set(level2_1_dict[key])) 
            temp_flags[level2_1_dict[key]] = 1
            temp_flags = np.multiply(temp_flags, map_flags).astype(bool)

            points = np.vstack([points, map_points[temp_flags]])
            colors = np.vstack([colors, map_colors[temp_flags]])
            map_flags[temp_flags] = 0
            # self.ar_flags[temp_flags] = level
            if id_flags is not None:
                flags = np.vstack([flags, id_flags[temp_flags].reshape(-1, 1)])

        # if flag:
        #     msg = self.xyzrgb_array_to_pointcloud2(points + human_center , colors, stamp=rospy.get_rostime(), frame_id='test', seq=None)
        #     self.pub_test_temp.publish(msg)
        #     self.br.sendTransform((0, 0, 0),
        #                         tf.transformations.quaternion_from_euler(0, 0, 0),
        #                         rospy.Time.now(),
        #                         'test',
        #                         'map')
        return points, colors, map_flags, flags

    def rot_ar(self, sphere, frame_id):
        try:
            (trans, rot) = self.tflistener.lookupTransform('map', frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        r_ar = R.from_quat(rot)
        sphere_points = sphere
        ar_points = r_ar.apply(sphere_points) + np.array(trans)
        return ar_points


    def check_visual(self, map_points, map_colors, id_flags):
        print('map_visual_check')

        self.br.sendTransform((5, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'map_in_human',
                            'human_frame')

        z_min, z_max, y_min, y_max = -0.5, 0.5, -1, 1
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
        r_ar = R.from_quat(rot_ar_in_h)
        plane_in_h = r_ar.apply(stdplane) + np.array(trans_ar_in_h)
        map_flags = np.ones(map_points.shape[0])
        points, colors, map_flags, flags = self.check_visual_level([0], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        colors[:, 0] = colors[:, 0]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])   
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-40], 'y', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 1)
        colors[:, 0] = colors[:, 0]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        self.ar_flags[:total_points.shape[0]] = 1
        temp_shape = total_points.shape[0]

        #level2
        points, colors, map_flags, flags = self.check_visual_level([50], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2, flag = True)
        colors[:, 1] = colors[:, 1]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-50], 'z', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, 1] = colors[:, 1]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([50, -40], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, 1] = colors[:, 1]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 

        points, colors, map_flags, flags = self.check_visual_level([-50, -40], 'zy', map_points, map_colors, plane_in_h, human_center, trans_ar_in_h, map_flags, id_flags, level = 2)
        colors[:, 1] = colors[:, 1]*0.25
        total_points = np.vstack([total_points, points])
        total_colors = np.vstack([total_colors, colors])
        total_flags = np.vstack([total_flags, flags]) 
        self.ar_flags[temp_shape:total_points.shape[0]] = 2
        temp_shape = total_points.shape[0]

        #level3
        map_flags = map_flags.astype(bool)
        colors = map_colors[map_flags]
        colors[:, 2] = colors[:, 2]*0.25
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
        msg = self.pointcloud2withrgb(total_points + human_center, total_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
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
        map_points = self.map_points
        map_colors = self.map_colors
        flags = np.zeros(3)
        bunny_points = self.rot_ar(self.bunny_points, 'bunny_points')
        if bunny_points is not None:
            map_points = np.vstack([map_points, bunny_points])
            map_colors = np.vstack([map_colors, self.bunny_colors])
            flags[0] = 1
        table_points = self.rot_ar(self.table_points, 'table_points')
        if table_points is not None:
            map_points = np.vstack([map_points, table_points])
            map_colors = np.vstack([map_colors, self.table_colors])
            flags[1] = 1
        teapot_points = self.rot_ar(self.teapot_points, 'teapot_points')
        if teapot_points is not None:
            map_points = np.vstack([map_points, teapot_points])
            map_colors = np.vstack([map_colors, self.teapot_colors])
            flags[2] = 1

        # try:
        #     (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return 

        # try:
        #     (trans_bunny, rot_bunny) = self.tflistener.lookupTransform('map', 'bunny_points', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return 
        # try:
        #     (trans_table, rot_table) = self.tflistener.lookupTransform('map', 'table_points', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return 
        # try:
        #     (trans_teapot, rot_teapot) = self.tflistener.lookupTransform('map', 'teapot_points', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     return 

        # id_flags = None
        # if np.sum(flags) == 3:
        #     id_flags = np.zeros(map_points.shape[0])
        #     id_flags[:self.map_points.shape[0]] = 0
        #     id_flags[self.map_points.shape[0] : self.map_points.shape[0] + self.bunny_points.shape[0]] = 1
        #     id_flags[self.map_points.shape[0] + self.bunny_points.shape[0]: self.map_points.shape[0] + self.bunny_points.shape[0] + self.table_points.shape[0]] = 2
        #     id_flags[self.map_points.shape[0] + self.bunny_points.shape[0] + self.table_points.shape[0]:] = 3
        #     # self.pub_id_flags.publish(id_flags.astype(np.float32))
        
        # if self.display_points is None:
        #     self.frame_trans = np.vstack([np.array(trans_h),np.array(trans_bunny), np.array(trans_table), np.array(trans_teapot)]) 
        #     self.frame_rot = np.array(rot_h)
        #     self.ar_flags = np.zeros(map_points.shape[0])
        #     self.check_visual(map_points, map_colors, id_flags)
        # else:
        #     dist_trans = np.linalg.norm(self.frame_trans - np.vstack([np.array(trans_h),np.array(trans_bunny), np.array(trans_table), np.array(trans_teapot)]))
        #     dist_rot = np.linalg.norm(self.frame_rot - np.array(rot_h))
        #     if dist_trans > 0.5 or dist_rot > 0.5 or self.ar_flags.shape[0] != map_points.shape[0]:
        #         # print(dist_trans, dist_rot)
        #         # print(self.ar_flags.shape[0], map_points.shape[0])
        #         self.ar_flags = np.zeros(map_points.shape[0])
        #         self.check_visual(map_points, map_colors, id_flags)
        #     else:
        msg = self.pointcloud2withrgb(map_points, map_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
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
            mapper.bunny_mutex.acquire()
            mapper.teapot_mutex.acquire()
            mapper.table_mutex.acquire()
            mapper.map_mutex.acquire()
            
            if (mapper.bunny_points is not None) and (mapper.table_points is not None) and (mapper.teapot_points is not None) and (mapper.map_points is not None):
                mapper.callback()
                mapper.bunny_mutex.release()
                mapper.teapot_mutex.release()
                mapper.table_mutex.release()
                mapper.map_mutex.release()
            else:
                mapper.bunny_mutex.release()
                mapper.teapot_mutex.release()
                mapper.table_mutex.release()
                mapper.map_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
    

