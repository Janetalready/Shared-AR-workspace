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

class PerceptualMap:
    def __init__(self):
        rospy.init_node('voxel_listener', anonymous=True)
        self.tflistener = tf.TransformListener()
        self.buuny_sub = rospy.Subscriber('bunny_world', PointCloud2, self.bunny_callback, queue_size=1)
        self.teapot_sub = rospy.Subscriber('teapot_world', PointCloud2, self.teapot_callback, queue_size=1)
        self.table_sub = rospy.Subscriber('table_world', PointCloud2, self.table_callback, queue_size=1)
        self.map_sub = rospy.Subscriber('rtabmap/cloud_map', PointCloud2, self.map_callback, queue_size=1)
        self.br = tf.TransformBroadcaster()
        self.pub_test = rospy.Publisher('visual_map', PointCloud2, queue_size=1)
        self.pub_test_temp = rospy.Publisher('test_table', PointCloud2, queue_size=1)

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

    def check_occlusion(self, frame_id, sphere, colors, map_points, map_colors):
        print(frame_id)

        try:
            (trans, rot) = self.tflistener.lookupTransform('map', frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return map_points, map_colors

        self.br.sendTransform((1, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'ar_in_human',
                            'human_frame')
        
        ar_center = np.array(trans)

        try:
            (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return map_points, map_colors

        r_ar = R.from_quat(rot)
        sphere_points = sphere
        ar_points = r_ar.apply(sphere_points) + np.array(trans)

        human_center = np.array(trans_h)

        try:
            (trans_ar_in_h, rot_ar_in_h) = self.tflistener.lookupTransform('map', 'ar_in_human', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return map_points, map_colors 
        
        position_in_h = np.array(trans_ar_in_h)
        
        # #shift to origin
        ar_center = ar_center - human_center
        position_in_h = position_in_h - human_center
        slam_points = map_points - human_center
        ar_points = ar_points - human_center
        cos_angles = ar_points.dot(position_in_h)

            
        assert colors.shape == ar_points.shape
        cam_center_fix = human_center
        human_center = np.array([0, 0, 0])

        theta1, theta2 = self.cal_quaternion(ar_center)
        print(theta1, theta2)
        r = R.from_euler('zy', [theta1, theta2])
        slam_points_rot = r.apply(slam_points)
        ar_points_temp = ar_points
        ar_points_rot = r.apply(ar_points)

        theta1, theta2 = self.cal_quaternion(position_in_h)
        r = R.from_euler('z', -theta1)
        slam_points_rot = r.apply(slam_points)
        ar_points_rot = r.apply(ar_points)
        colors = colors[ar_points_rot[:, 0]>=0]
        ar_points = ar_points[ar_points_rot[:, 0]>=0]
        slam_points = slam_points[slam_points_rot[:, 0]>=0]
        slam_points_rot = slam_points_rot[slam_points_rot[:, 0]>=0]
        ar_points_rot = ar_points_rot[ar_points_rot[:, 0]>=0]
        distance = np.mean(ar_points_rot[:, 0])

        if ar_points.shape[0] == 0:
            return map_points, map_colors

        z_min = np.min(ar_points_rot[:, 2])
        z_max = np.max(ar_points_rot[:, 2])
        y_min = np.min(ar_points_rot[:, 1])
        y_max = np.max(ar_points_rot[:, 1])

        candidate_points_id = np.logical_and(np.logical_and(np.logical_and(np.logical_and(slam_points_rot[:, 0] <= distance, slam_points_rot[:, 1] <= y_max), slam_points_rot[:, 1]>=y_min), 
                                slam_points_rot[:, 2]<=z_max), slam_points_rot[:, 2]>=z_min)
        candidate_points = slam_points_rot[candidate_points_id]

        if candidate_points.shape[0] > 0:
            ar_points_rot_t = ar_points_rot
            candidate_points = np.round(candidate_points, decimals = 2)
            ar_points_rot_t = np.round(ar_points_rot_t, decimals = 2)
            
            flags = []
            for pid, ar_point in enumerate(ar_points_rot_t[:, 1:]):
                dist = np.linalg.norm(ar_point - candidate_points[:, 1:], axis = 1)
                if dist[dist <= 0.02].shape[0] == 0:
                    flags.append(pid)

            # remove_id = np.logical_and(np.absolute(candidate_points[:, 1] - ar_points_rot_t[:, 1]) < 0.02, np.absolute(candidate_points[:, 2] - ar_points_rot_t[:, 2]) < 0.02)

            ar_real = ar_points[flags] + cam_center_fix
            colors_t = colors[flags]
        else:
            ar_real = ar_points + cam_center_fix
            colors_t = colors
        
        new_map_points = np.vstack([map_points, ar_real])
        new_map_colors = np.vstack([map_colors, colors_t])

        # if frame_id == 'table_points':
        #     msg = self.xyzrgb_array_to_pointcloud2(ar_points_rot + cam_center_fix, colors_temp, stamp=rospy.get_rostime(), frame_id='test', seq=None)
        #     self.pub_test_temp.publish(msg)
        #     self.br.sendTransform((0, 0, 0),
        #                         tf.transformations.quaternion_from_euler(0, 0, 0),
        #                         rospy.Time.now(),
        #                         'test',
        #                         'map')
        return new_map_points, new_map_colors
    
    def check_visual_level(self, rot_angle, map_points, plane_in_h, human_center, map_colors, trans_ar_in_h, flag=False):
        plane_in_h = plane_in_h - human_center
        map_points = map_points - human_center
        trans_ar_in_h = trans_ar_in_h - human_center

        theta1, theta2 = self.cal_quaternion(trans_ar_in_h)
        rot = R.from_euler('z', -theta1)
        map_rot = rot.apply(map_points)
        plane_in_h = rot.apply(plane_in_h)

        rot = R.from_euler('z', rot_angle, degrees=True)
        map_rot = rot.apply(map_rot)
        y_min = np.min(plane_in_h[:, 1])
        y_max = np.max(plane_in_h[:, 1])
        z_min = np.min(plane_in_h[:, 2])
        z_max = np.max(plane_in_h[:, 2])

        candidate_points_id = np.logical_and(np.logical_and(np.logical_and(np.logical_and(map_rot[:, 1] <= y_max, map_rot[:, 1]>=y_min), 
                                map_rot[:, 2]<=z_max), map_rot[:, 2]>=z_min), map_rot[:, 0] >= 0.3)
        candidate_points_ids = np.where(candidate_points_id)
        negative_ids = np.where(~candidate_points_id)
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
                indx = int(y_idx*100 + z_idx)
                if indx in level2_1_dict:
                    if map_points[level2_1_dict[indx]][0] > x:
                        level2_1_dict[indx] = pid
                else:
                    level2_1_dict[indx] = pid
                # candidate_point = map_rot[pid][1:]
                # dist = np.linalg.norm(candidate_point - plane_in_h[:, 1:], axis = 1)
                # idx = np.argmin(dist)
                # if idx in level2_1_dict:
                #     if candidate_point[0] < map_rot[level2_1_dict[idx]][0]:
                #         level2_1_dict[idx] = pid
                # else:
                #     level2_1_dict[idx] = pid
        return level2_1_dict.values(), candidate_points_ids[0], negative_ids[0]

    def check_visual(self, map_points, map_colors):
        print('map_visual_check')

        self.br.sendTransform((5, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'map_in_human',
                            'human_frame')

        z_min, z_max, y_min, y_max = -1, 1, -1, 1
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

        r_ar = R.from_quat(rot_ar_in_h)
        plane_in_h = r_ar.apply(stdplane) + np.array(trans_ar_in_h)
        # else:
        #level1
        color_ids, candidate_ids, negative_ids = self.check_visual_level(0, map_points, plane_in_h, human_center, map_colors, trans_ar_in_h, flag=True)
        colors_level1 = map_colors[color_ids]
        colors_level1[:, 0] = colors_level1[:, 0]*0.25
        map_level1 = map_points[color_ids]
        map_level3 = map_points[negative_ids]
        colors_level3 = map_colors[negative_ids]
        map_points_new = np.delete(map_points, candidate_ids, axis = 0)
        map_colors_new = np.delete(map_colors, candidate_ids, axis = 0)
        # msg = self.xyzrgb_array_to_pointcloud2(map_points[color_ids], map_colors[color_ids], stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
        # self.pub_test.publish(msg)
        # self.br.sendTransform((0, 0, 0),
        #                     tf.transformations.quaternion_from_euler(0, 0, 0),
        #                     rospy.Time.now(),
        #                     'visual_map',
        #                     'map')
            
        #level2
        color_ids, candidate_ids, negative_ids = self.check_visual_level(2*np.arctan2(1, 5)/math.pi*180, map_points_new, plane_in_h, human_center, map_colors, trans_ar_in_h) 
        colors_level21 = map_colors[color_ids]
        colors_level21[:, 1] = colors_level21[:, 1]*0.25
        map_level21 = map_points[color_ids]
        map_level3 = np.vstack([map_level3, map_points_new[negative_ids]])
        colors_level3 = np.vstack([colors_level3, map_colors_new[negative_ids]])
        map_points_new = np.delete(map_points_new, candidate_ids, axis = 0)
        map_colors_new = np.delete(map_colors_new, candidate_ids, axis = 0)

        color_ids, candidate_ids, negative_ids = self.check_visual_level(-2*np.arctan2(1, 5)/math.pi*180, map_points_new, plane_in_h, human_center, map_colors, trans_ar_in_h)
        colors_level22 = map_colors[color_ids]
        colors_level22[:, 1] = colors_level22[:, 1]*0.25
        map_level22 = map_points[color_ids]
        map_level3 = np.vstack([map_level3, map_points_new[negative_ids]])
        colors_level3 = np.vstack([colors_level3, map_colors_new[negative_ids]])
        map_points_new = np.delete(map_points_new, candidate_ids, axis = 0)
        map_colors_new = np.delete(map_colors_new, candidate_ids, axis = 0)

        color_ids, candidate_ids, negative_ids = self.check_visual_level(4*np.arctan2(1, 5)/math.pi*180, map_points_new, plane_in_h, human_center, map_colors, trans_ar_in_h, flag=True)
        colors_level23 = map_colors[color_ids]
        colors_level23[:, 1] = colors_level23[:, 1]*0.25
        map_level23 = map_points[color_ids]
        map_level3 = np.vstack([map_level3, map_points_new[negative_ids]])
        colors_level3 = np.vstack([colors_level3, map_colors_new[negative_ids]])
        map_points_new = np.delete(map_points_new, candidate_ids, axis = 0)
        map_colors_new = np.delete(map_colors_new, candidate_ids, axis = 0)
        # msg = self.xyzrgb_array_to_pointcloud2(map_points[color_ids], map_colors[color_ids], stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
        # self.pub_test.publish(msg)
        # self.br.sendTransform((0, 0, 0),
        #                     tf.transformations.quaternion_from_euler(0, 0, 0),
        #                     rospy.Time.now(),
        #                     'visual_map',
        #                     'map')

        color_ids, candidate_ids, negative_ids = self.check_visual_level(-4*np.arctan2(1, 5)/math.pi*180, map_points_new, plane_in_h, human_center, map_colors, trans_ar_in_h)
        colors_level24 = map_colors[color_ids]
        colors_level24[:, 1] = colors_level24[:, 1]*0.25
        map_level24 = map_points[color_ids]
        map_level3 = np.vstack([map_level3, map_points_new[negative_ids]])
        colors_level3 = np.vstack([colors_level3, map_colors_new[negative_ids]])
        map_points_new = np.delete(map_points_new, candidate_ids, axis = 0)
        map_colors_new = np.delete(map_colors_new, candidate_ids, axis = 0)
        
        colors_level3[:, 2] = colors_level3[:, 2]*0.25
        # self.total_maps = np.vstack([map_level1, map_level21, map_level22, map_level23, map_level24, map_level3])
        # self.total_colors = np.vstack([colors_level1, colors_level21, colors_level22, colors_level23, colors_level24, colors_level3])
        self.total_maps = map_level1
        self.total_colors = colors_level1
        msg = self.xyzrgb_array_to_pointcloud2(self.total_maps, self.total_colors, stamp=rospy.get_rostime(), frame_id='visual_map', seq=None)
        self.pub_test.publish(msg)
        self.br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'visual_map',
                            'map')
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
        map_points, map_colors = self.check_occlusion('bunny_points', self.bunny_points, self.bunny_colors, self.map_points, self.map_colors)
        map_points, map_colors = self.check_occlusion('teapot_points', self.teapot_points, self.teapot_colors, map_points, map_colors)
        map_points, map_colors = self.check_occlusion('table_points', self.table_points, self.table_colors, map_points, map_colors)
        self.check_visual(map_points, map_colors)

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
    

