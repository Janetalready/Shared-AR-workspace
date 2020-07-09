#!/usr/bin/env python
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point, PointStamped, Twist
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
from nav_msgs.msg import OccupancyGrid
from rospy.numpy_msg import numpy_msg
from threading import Thread, Lock
from rospy_tutorials.msg import Floats
from astar import *
import time
import pickle
from vel_commandv2 import *

class GridMap:
    def __init__(self):
        rospy.init_node('grid_map', anonymous=True)
        self.br = tf.TransformBroadcaster()
        self.tflistener = tf.TransformListener()
        self.map_points_sub = rospy.Subscriber('visual_map', PointCloud2, self.map_callback, queue_size=1)
        self.grid_map_sub = rospy.Subscriber('projected_map', OccupancyGrid, self.grid_callback, queue_size=1)
        self.ar_flag_sub = rospy.Subscriber("ar_flags", numpy_msg(Floats), self.ar_callback, queue_size=1)
        self.id_flag_sub = rospy.Subscriber("id_flags", numpy_msg(Floats), self.id_callback, queue_size=1)
        self.grid_pub = rospy.Publisher('navigate_grid', OccupancyGrid, queue_size=5)
        self.new_grid_pub = rospy.Publisher('new_grid', OccupancyGrid, queue_size=5)
        self.map_points = None
        self.grid_map = None
        self.ar_flags = None
        self.id_flags = None
        self.map_mutex = Lock()
        self.grid_mutex = Lock()
        self.ar_mutex = Lock()
        self.id_mutex = Lock()

        self.level3_flags = None
        self.begin_check = 0
        self.save = 1

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

    def map_callback(self, data):
        self.map_mutex.acquire()
        map_points = list(pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True))
        self.map_mutex.release()
        self.map_points = np.array(map_points)

    def grid_callback(self, data):
        self.grid_mutex.acquire()
        self.grid_map = data
        self.grid_mutex.release()

    def ar_callback(self, data):
        self.ar_mutex.acquire()
        if self.ar_flags is None or np.linalg.norm(self.ar_flags - data.data) > 0:
            self.begin_check = 1
            self.ar_flags = data.data
        self.ar_mutex.release()

    def id_callback(self, data):
        self.id_mutex.acquire()
        self.id_flags = data.data.reshape(-1)
        self.id_mutex.release()

    def check_nearest_obj(self, obj_list, src, r, grid_map):
        min_dist = 10000000
        min_path = None
        for obj_idx, i in enumerate(obj_list):
            try:
                (trans_ar, rot_ar) = self.tflistener.lookupTransform('map', self.frame_ids[i], rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return [obj_idx, None, None, i]
            trans_ar = np.array(trans_ar)
            trans_ar = r.apply(trans_ar)
            indx = int((trans_ar[0] - self.grid_map.info.origin.position.x)/self.grid_map.info.resolution)
            indy = int((trans_ar[1] - self.grid_map.info.origin.position.y)/self.grid_map.info.resolution)
            shifts = list(product(np.arange(-2, 2, 1), np.arange(-2, 2, 1)))
            for shift_x, shift_y in shifts:
                dst = (indx + shift_x, indy + shift_y)
                # grid_map[grid_map == 0] = 1
                grid_map[grid_map == -1] = 0
                grid_map[grid_map > 0] = 1
                planner = AstarPlanner(grid=grid_map)
                path, cost = planner.search(src, dst)
                print(cost)
                if cost < min_dist and len(path) > 0:
                    min_dist = cost
                    min_path = [obj_idx, path, dst, i]
        return min_path

    def pub_pose(self, stop, frame_id):
        indx = stop[0]
        indy = stop[1]
        origin_x = self.grid_map.info.origin.position.x
        origin_y = self.grid_map.info.origin.position.y
        resolution = self.grid_map.info.resolution

        p = PoseStamped()
        p.header.frame_id = self.command_frame[frame_id]
        p.header.stamp = rospy.Time.now()

        quat = self.grid_map.info.origin.orientation
        print(quat)
        if quat.w == 0:
            quat.w = 1
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        r = r.inv()
        r = r.as_quat()

        p.pose.position.x = stop[0]*resolution + origin_x
        p.pose.position.y = stop[1]*resolution + origin_y
        p.pose.position.z = 0.02
        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 1

        self.br.sendTransform((stop[0]*resolution + origin_x, stop[1]*resolution + origin_y, 0.22),
                                [0, 0, 0, 1],
                                rospy.Time.now(),
                                self.command_frame[frame_id],
                                'map')
        self.compub[frame_id].publish(p)

    def find_nearest_indx(self, dst, indx):
        dist = np.linalg.norm(indx - dst, axis = 1)
        idx = np.argsort(dist)[:10]
        return indx[idx], idx

    def find_neareat_point_inlevel1(self, dst, map_points, r, origin):
        map_points = r.apply(map_points)
        map_points = map_points[:, :2]
        indx = (map_points - origin)/self.grid_map.info.resolution
        return self.find_nearest_indx(dst, indx)

    def find_highest_cost_obj(self, obj_list, src, r, grip_map, human_pos):
        min_dist = 10000000
        min_path = None
        max_cos = 3.14/2
        max_trans = None
        max_obj_id = None
        max_i = None
        angles = []
        trans = []
        for obj_idx, i in enumerate(obj_list):
            try:
                (trans_ar, rot_ar) = self.tflistener.lookupTransform('human_frame', self.frame_ids[i], rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return None
            trans_ar = np.array(trans_ar)
            if np.linalg.norm(trans_ar) > 0:
                trans_ar = trans_ar/np.linalg.norm(trans_ar)
            angle = trans_ar.dot(np.array([1, 0, 0]))
            angles.append(angle)
            trans.append(trans_ar)

        idx = np.argsort(np.array(angles))
        for obj_idx in idx:
            max_trans = trans[obj_idx]
            max_trans = r.apply(max_trans)
            indx = int((max_trans[0] - self.grid_map.info.origin.position.x)/self.grid_map.info.resolution)
            indy = int((max_trans[1] - self.grid_map.info.origin.position.y)/self.grid_map.info.resolution)
            shifts = list(product(np.arange(-2, 2, 1), np.arange(-2, 2, 1)))
            for shift_x, shift_y in shifts:
                dst = (indx + shift_x, indy + shift_y)
                grid_map[grid_map == -1] = 0
                grid_map[grid_map > 0] = 1
                planner = AstarPlanner(grid=grid_map)
                path, cost = planner.search(src, dst)
                if cost < min_dist and len(path) > 0:
                    min_dist = cost
                    min_path = [max_obj_id, path, dst, max_i]
            if min_path:
                return min_path
        return min_path

    def callback(self):
        print('points recevied')
        height = self.grid_map.info.height
        width = self.grid_map.info.width
        data = -1*np.ones(width*height)

        level3_objects = []
        level2_objects = []
        num_objs = np.unique(self.id_flags).shape[0]

        for i in range(1, num_objs):
            assert self.ar_flags.shape[0] == self.id_flags.shape[0] == self.map_points.shape[0]
            ar_flags = self.ar_flags[self.id_flags == i]
            if ar_flags[ar_flags == 3].shape[0] > ar_flags.shape[0]*0.9:
                level3_objects.append(i)
            else:
                level2_objects.append(i)
        
        if len(level3_objects) == 0:
            level3_objects = range(1, num_objs)
        print(level3_objects)
        try:
            (trans_r, rot_r) = self.tflistener.lookupTransform('map', 'base_footprint', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        try:
            (trans_h, rot_h) = self.tflistener.lookupTransform('map', 'human_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        trans_r = np.array(trans_r)
        quat = self.grid_map.info.origin.orientation
        if quat.w == 0:
            quat.w = 1
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        trans_r = r.apply(trans_r)
        origin_x = self.grid_map.info.origin.position.x
        origin_y = self.grid_map.info.origin.position.y
        src_x = int((trans_r[0] - origin_x)/self.grid_map.info.resolution)
        src_y = int((trans_r[1] - origin_y)/self.grid_map.info.resolution)

        level3_flags = np.zeros(len(level3_objects))
        grid_map = self.grid_map.data
        grid_map = np.array(grid_map).reshape((height, width))

        indx_h = int((trans_h[0] - origin_x)/self.grid_map.info.resolution)
        indy_h = int((trans_h[1] - origin_y)/self.grid_map.info.resolution)
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                grid_map[indx_h + i, indy_h + j] = 1

        objs_path = dict()
        objs_path['res'] = self.grid_map.info.resolution
        objs_path['origin'] = self.grid_map.info.origin
            

        while(len(level3_objects) > 0):
            if self.begin_check:
                self.begin_check = 0
                return
            path = self.find_highest_cost_obj(level3_objects, (src_x, src_y), r, grid_map)
            if path is None:
                print('cost is too high')
                return

            obj_idx = path[0]
            level3_flags[obj_idx] = 1
            del level3_objects[obj_idx]

            data = -1*np.ones(width*height)
            objs_path[path[3]] = []
            objs_path[path[3]].append(path[1])
            for stop in path[1]:
                stop = np.array(stop)
                src = np.array([src_x, src_y])
                dirt = stop - src
                src = stop
                data[width*stop[1] + stop[0]] = 100

            # #find nearest point in level 1 - 2
            cost = 1000000
            min_path = None

            # moving_stops, idxs = self.find_neareat_point_inlevel1(path[1][-1], map_points, r, np.array([origin_x, origin_y]))
            # grid_map[grid_map == -1] = 0
            # grid_map[grid_map > 0] = 1
            # objs_path[path[3]].append(moving_stops)

            indx_h = int(( [0] - origin_x)/self.grid_map.info.resolution)
            indy_h = int((trans_h[1] - origin_y)/self.grid_map.info.resolution)
            moving_stops = [(indx_h - 2, indy_h - 2), (indx_h - 2, indy_h - 1), (indx_h - 2, indy_h), (indx_h - 2, indy_h + 1), (indx_h - 2, indy_h + 2), 
                            (indx_h - 1, indy_h - 2), (indx_h - 1, indy_h + 2), (indx_h, indy_h - 2), (indx_h, indy_h + 2), (indx_h + 1, indy_h - 2),
                            (indx_h + 1, indy_h + 2), (indx_h + 2, indy_h - 2), (indx_h + 2, indy_h - 1), (indx_h + 2, indy_h), (indx_h + 2, indy_h + 1),
                            (indx_h + 2, indy_h + 2)]
            for pid, moving_stop in enumerate(moving_stops):            
                planner = AstarPlanner(grid=grid_map)
                planned_path, cost = planner.search(path[1][-1], (int(moving_stop[0]), int(moving_stop[1])))
                min_path = planned_path
                if min_path:
                    break
            if min_path is None:
                return
            objs_path[path[3]].append(min_path)

            #loop send vel cmd and update ar location
            for stop in min_path:
                stop = np.array(stop)
                src = path[1][-1]
                dirt = stop - src
                src = stop
                data[width*stop[1] + stop[0]] = 100
            msg = self.grid_map
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'ar_grid_map'
            msg.data = tuple(data)
            msg.info.width = self.grid_map.info.width
            msg.info.height = self.grid_map.info.height
            msg.info.origin = self.grid_map.info.origin
            self.grid_pub.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                [0, 0, 0, 1],
                                rospy.Time.now(),
                                "ar_grid_map",
                                'map')
            try:
                MoveBaseSeq()
            except rospy.ROSInterruptException:
                rospy.loginfo("Navigation finished.")

            

            self.begin_check = 0
            print('sent')
        if self.save:
            print('writing....')
            with open("/home/shuwen/pathv2.p", 'wb') as f:
                pickle.dump(objs_path, f)
            print("writing saved")


if __name__ == '__main__':
    mapper = GridMap()
    try:
        while not rospy.is_shutdown():
            mapper.map_mutex.acquire()
            mapper.grid_mutex.acquire()
            mapper.ar_mutex.acquire()
            
            if mapper.begin_check and (mapper.map_points is not None) and (mapper.grid_map is not None) and (mapper.id_flags is not None) and (mapper.ar_flags is not None and mapper.ar_flags.shape[0] == mapper.map_points.shape[0]):
                mapper.callback()
                mapper.map_mutex.release()
                mapper.grid_mutex.release()
                mapper.ar_mutex.release()
            else:
                mapper.map_mutex.release()
                mapper.grid_mutex.release()
                mapper.ar_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
    

