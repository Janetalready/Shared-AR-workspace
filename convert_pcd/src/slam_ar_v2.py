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

class SlamAR:
    def __init__(self):
        rospy.init_node('voxel_listener', anonymous=True)
        self.tflistener = tf.TransformListener()
        self.pub_now_virtual = dict()
        self.pub_total_virtual = dict()
        self.pub_total_virtual['table_points'] = rospy.Publisher('table_total_virtual', PointCloud2, queue_size=1)
        self.pub_total_virtual['bunny_points'] = rospy.Publisher('bunny_total_virtual', PointCloud2, queue_size=1)
        self.pub_total_virtual['teapot_points'] = rospy.Publisher('teapot_total_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['table_points'] = rospy.Publisher('table_now_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['bunny_points'] = rospy.Publisher('bunny_now_virtual', PointCloud2, queue_size=1)
        self.pub_now_virtual['teapot_points'] = rospy.Publisher('teapot_now_virtual', PointCloud2, queue_size=1)
        self.pub_test = rospy.Publisher('sphere_test', PointCloud2, queue_size=1)
        self.buuny_sub = rospy.Subscriber('bunny_world', PointCloud2, self.bunny_callback, queue_size=1)
        self.teapot_sub = rospy.Subscriber('teapot_world', PointCloud2, self.teapot_callback, queue_size=1)
        self.table_sub = rospy.Subscriber('table_world', PointCloud2, self.table_callback, queue_size=1)
        self.slam_sub = rospy.Subscriber('map_update', PointCloud2, self.slam_callback, queue_size=1)
        # pub = rospy.Publisher('voxel_update', PointCloud2, queue_size=5)
        self.br = tf.TransformBroadcaster()
        self.virtual_points = dict()
        self.virtual_colors = dict()
        self.new_points = None
        self.flag = {'table_points':1, 'bunny_points':1, 'teapot_points':1}
        self.current_map = None
        self.current_color = None

        self.bunny_mutex = Lock()
        self.teapot_mutex = Lock()
        self.table_mutex = Lock()
        self.slam_mutex = Lock()

        self.bunny_points = None
        self.bunny_colors = None
        self.teapot_points = None
        self.teapot_colors = None
        self.table_points = None
        self.table_colors = None


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

    def slam_callback(self, data):
        self.slam_mutex.acquire()
        points = []
        colors = []
        for idx, p in enumerate(pc2.read_points(data, field_names = ("x", "y", "z", "r", "g", "b"), skip_nans=True)):
                point = np.array([p[0],p[1],p[2]])
                point = np.array(point)
                points.append(point)
                colors.append([p[3],p[4],p[5]])
        self.slam_mutex.release()
        self.slam_points = np.array(points)
        self.slam_colors = np.array(colors)

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

    def check_ar(self, frame_id, sphere, colors, slam, slam_colors):
        print(frame_id)

        try:
            (trans, rot) = self.tflistener.lookupTransform('map', frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 

        self.br.sendTransform((1, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            'ar_in_cam',
                            'camera_mount_link')
        
        ar_center = np.array(trans)

        try:
            (trans_cam, rot_cam) = self.tflistener.lookupTransform('map', 'camera_mount_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        r_ar = R.from_quat(rot)
        sphere_points = sphere
        ar_points = r_ar.apply(sphere_points) + np.array(trans)

        map_points = slam
        map_colors = slam_colors
        cam_center = np.array(trans_cam)

        try:
            (trans_ar_in_cam, rot_ar_in_cam) = self.tflistener.lookupTransform('map', 'ar_in_cam', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return 
        
        position_in_cam = np.array(trans_ar_in_cam)
        
        # #shift to origin
        ar_center = ar_center - cam_center
        position_in_cam = position_in_cam - cam_center
        slam_points = map_points - cam_center
        ar_points = ar_points - cam_center
        cos_angles = ar_points.dot(position_in_cam)
        colors = colors[cos_angles>=0]
        ar_points = ar_points[cos_angles>=0]
        cos_angles = slam_points.dot(position_in_cam)
        current_map_colors = map_colors[cos_angles>=0]
        current_map = map_points[cos_angles>=0]
        slam_points = slam_points[cos_angles>=0]

            
        assert colors.shape == ar_points.shape
        cam_center_fix = cam_center
        cam_center = np.array([0, 0, 0])

        theta1, theta2 = self.cal_quaternion(ar_center)
        print(theta1, theta2)
        r = R.from_euler('zy', [-theta1, theta2])
        slam_points_rot = r.apply(slam_points)
        ar_points_rot = r.apply(ar_points)
        distance = np.mean(ar_points_rot[:, 0])

        if ar_points.shape[0] == 0:
            if self.flag[frame_id]:
                ar_real = np.array([0, 0, 0])
                colors_t = np.array([0, 0, 0])
                self.virtual_points[frame_id] = ar_real
                self.virtual_colors[frame_id] = colors_t
                self.flag[frame_id] = 0
            else:
                self.virtual_points[frame_id] = np.vstack([self.virtual_points[frame_id], ar_real])
                self.virtual_colors[frame_id] = np.vstack([self.virtual_colors[frame_id], colors_t])

            msg = self.xyzrgb_array_to_pointcloud2(ar_real, colors_t, stamp=rospy.get_rostime(), frame_id=frame_id + '_now', seq=None)
            self.pub_now_virtual[frame_id].publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                frame_id + '_now',
                                'map')
            
            colors = self.virtual_colors[frame_id]

            msg = self.xyzrgb_array_to_pointcloud2(self.virtual_points[frame_id], colors, stamp=rospy.get_rostime(), frame_id=frame_id + '_total', seq=None)
            self.pub_total_virtual[frame_id].publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                frame_id + '_total',
                                'map')
            return np.array([0, 0, 0]), np.array([0, 0, 0]), current_map, current_map_colors
        z_min = np.min(ar_points_rot[:, 2])
        z_max = np.max(ar_points_rot[:, 2])
        y_min = np.min(ar_points_rot[:, 1])
        y_max = np.max(ar_points_rot[:, 1])

        candidate_points_id = np.logical_and(np.logical_and(np.logical_and(np.logical_and(slam_points_rot[:, 0] <= distance, slam_points_rot[:, 1] <= y_max), slam_points_rot[:, 1]>=y_min), 
                                slam_points_rot[:, 2]<=z_max), slam_points_rot[:, 2]>=z_min)
        candidate_points = slam_points_rot[candidate_points_id]

        if candidate_points.shape[0] > 0:
            ar_points_rot_t = ar_points_rot[ar_points_rot[:, 0] <= distance]
            candidate_points = np.round(candidate_points, decimals = 2)
            ar_points_rot_t = np.round(ar_points_rot_t, decimals = 2)
            
            flags = []
            for pid, ar_point in enumerate(ar_points_rot_t[:, 1:]):
                dist = np.linalg.norm(ar_point - candidate_points[:, 1:], axis = 1)
                if dist[dist <= 0.02].shape[0] == 0:
                    flags.append(pid)

            # remove_id = np.logical_and(np.absolute(candidate_points[:, 1] - ar_points_rot_t[:, 1]) < 0.02, np.absolute(candidate_points[:, 2] - ar_points_rot_t[:, 2]) < 0.02)

            ar_points = ar_points[ar_points_rot[:, 0] <= distance]
            colors = colors[ar_points_rot[:, 0] <= distance]
            ar_real = ar_points[flags] + cam_center_fix
            colors_t = colors[flags]
        else:
            ar_points = ar_points[ar_points_rot[:, 0] <= distance]
            colors = colors[ar_points_rot[:, 0] <= distance]
            ar_real = ar_points + cam_center_fix
            colors_t = colors
        print(ar_real.shape)
        print(colors_t.shape)

        if self.flag[frame_id]:
            self.virtual_points[frame_id] = ar_real
            self.virtual_colors[frame_id] = colors_t
            self.flag[frame_id] = 0
        else:
            self.virtual_points[frame_id] = np.vstack([self.virtual_points[frame_id], ar_real])
            self.virtual_colors[frame_id] = np.vstack([self.virtual_colors[frame_id], colors_t])

        msg = self.xyzrgb_array_to_pointcloud2(ar_real, colors_t, stamp=rospy.get_rostime(), frame_id=frame_id + '_now', seq=None)
        self.pub_now_virtual[frame_id].publish(msg)
        self.br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            frame_id + '_now',
                            'map')
        
        colors = self.virtual_colors[frame_id]

        msg = self.xyzrgb_array_to_pointcloud2(self.virtual_points[frame_id], colors, stamp=rospy.get_rostime(), frame_id=frame_id + '_total', seq=None)
        self.pub_total_virtual[frame_id].publish(msg)
        self.br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            frame_id + '_total',
                            'map')
        
        

        # step = 0.02
        # coordinates = list(product([distance], np.arange(y_min, y_max, step), np.arange(z_min, z_max, step))) 
        # std_plane = np.array(coordinates)

        # ar_points_rot = ar_points_rot[ar_points_rot[:, 0] <= distance]
        # show_points = []
        # prj_dis = dict()
        # curr_position = np.array([0, 0, 0])
        # for pid, std_point in enumerate(std_plane):
        #     flag = 0
        #     for candidate in candidate_points:
        #         dist = np.linalg.norm(candidate)
        #         if self.cos_angle(candidate, std_point, curr_position) > 0.998 and dist < np.linalg.norm(std_point - curr_position):
        #             prj_dis[pid] = dist
        #             flag = 1
        #             break
        #     if flag == 0:
        #         prj_dis[pid] = 1000
        
        # for pid, std_point in enumerate(std_plane):
        #     for i in range(0, ar_points_rot.shape[0], 10):
        #     # for virtual_point in ar_points_rot:
        #         virtual_point = ar_points_rot[i]
        #         dist = np.linalg.norm(virtual_point - curr_position)
        #         if self.cos_angle(virtual_point, std_point, curr_position) > 0.998 and dist < prj_dis[pid]:
        #             show_points.append(virtual_point)
        # show_points = np.array(show_points)
        # print(show_points.shape)

        # test_points = np.vstack([candidate_points, ar_points_test])
        # colors = np.ones(ar_points.shape)
        # if len(colors.shape) == 1:
        #     colors = colors.reshape((1, -1))
        # if frame_id == 'cube_points':
        #     colors[:, 0] = 255.0/255
        #     colors[:, 1] = 192.0/255
        #     colors[:, 2] = 203.0/255
        # elif frame_id == 'sphere2_points':
        #     colors[:, 0] = 1
        #     colors[:, 1] = 1
        #     colors[:, 2] = 1
        # else:
        #     colors[:, 0] = 1
        #     colors[:, 1] = 1
        #     colors[:, 2] = 0

        # msg = self.xyzrgb_array_to_pointcloud2(current_map, current_map_colors, stamp=rospy.get_rostime(), frame_id='current_map', seq=None)
        # self.pub_test.publish(msg)
        # # self.br.sendTransform((trans[0], trans[1], trans[2]),
        # #                     (rot[0], rot[1], rot[2], rot[3]),
        # #                     rospy.Time.now(),
        # #                     frame_id + '_test',
        # #                     'map')
        # self.br.sendTransform((0, 0, 0),
        #                     tf.transformations.quaternion_from_euler(0, 0, 0),
        #                     rospy.Time.now(),
        #                     'current_map',
        #                     'map')
        return ar_real, colors_t, current_map, current_map_colors

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
        flag1 = self.check_ar('bunny_points', self.bunny_points, self.bunny_colors, self.slam_points, self.slam_colors)
        flag2 = self.check_ar('teapot_points', self.teapot_points, self.teapot_colors, self.slam_points, self.slam_colors)
        flag3 = self.check_ar('table_points', self.table_points, self.table_colors, self.slam_points, self.slam_colors)
        
        if flag1 is not None and flag2 is not None and flag3 is not None:
            bunny, colors_bunny, current_map, current_map_colors = flag1
            teapot, colors_teapot, current_map, current_map_colors = flag2
            table, colors_table, current_map, current_map_colors = flag3
            current_total = np.vstack([current_map, bunny, teapot, table])
            color_total = np.vstack([current_map_colors, colors_bunny, colors_teapot, colors_table])
            self.current_map = current_total
            self.current_color = color_total
            msg = self.pointcloud2withrgb(current_total, color_total, stamp=rospy.get_rostime(), frame_id='current_map', seq=None)
            # msg = self.xyzrgb_array_to_pointcloud2(current_total, color_total, stamp=rospy.get_rostime(), frame_id='current_map', seq=None)
            self.pub_test.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                'current_map',
                                'map')
        elif self.current_map is not None:
            msg = self.pointcloud2withrgb(self.current_map, self.current_color, stamp=rospy.get_rostime(), frame_id='current_map', seq=None)
            # msg = self.xyzrgb_array_to_pointcloud2(self.current_map, self.current_color, stamp=rospy.get_rostime(), frame_id='current_map', seq=None)
            self.pub_test.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                'current_map',
                                'map')

    def listener(self):
        bunny_points_sub = message_filters.Subscriber('bunny_world', PointCloud2)
        teapot_points_sub = message_filters.Subscriber('teapot_world', PointCloud2)
        table_points_sub = message_filters.Subscriber('table_world', PointCloud2)
        slam_points_sub = message_filters.Subscriber('map_update', PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([bunny_points_sub, teapot_points_sub, table_points_sub, slam_points_sub], 1, 100, allow_headerless=True)
        ts.registerCallback(self.callback)
        rospy.spin()

if __name__ == '__main__':
    slam_ar = SlamAR()
    try:
        while not rospy.is_shutdown():
            slam_ar.bunny_mutex.acquire()
            slam_ar.teapot_mutex.acquire()
            slam_ar.table_mutex.acquire()
            slam_ar.slam_mutex.acquire()
            
            if (slam_ar.bunny_points is not None) and (slam_ar.table_points is not None) and (slam_ar.teapot_points is not None) and (slam_ar.slam_points is not None):
                slam_ar.callback()
                slam_ar.bunny_mutex.release()
                slam_ar.teapot_mutex.release()
                slam_ar.table_mutex.release()
                slam_ar.slam_mutex.release()
            else:
                slam_ar.bunny_mutex.release()
                slam_ar.teapot_mutex.release()
                slam_ar.table_mutex.release()
                slam_ar.slam_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
    

