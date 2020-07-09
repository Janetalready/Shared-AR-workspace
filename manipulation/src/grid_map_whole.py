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

        self.frame_ids = {2: 'bottle_points', 3:'meat_points', 4:'potato_points', 5:'tomato_points', 6:'tomato2_points', 7:'veggie_points'}
        self.command_frame = {'bottle_points':'BottleCommand_ros', 'meat_points':'MeatCommand_ros', 'potato_points':'PotatoCommand_ros', 'tomato_points':'TomatoCommand_ros',\
                                'tomato2_points':'Tomato2Command_ros', 'veggie_points':'VeggieCommand_ros'}

        self.bottle_compub = rospy.Publisher('BottleCommand_ros', PoseStamped, queue_size=1)
        self.meat_compub = rospy.Publisher('MeatCommand_ros', PoseStamped, queue_size=1)
        self.potato_compub = rospy.Publisher('PotatoCommand_ros', PoseStamped, queue_size=1)
        self.tomato_compub = rospy.Publisher('TomatoCommand_ros', PoseStamped, queue_size=1)
        self.tomato2_compub = rospy.Publisher('Tomato2Command_ros', PoseStamped, queue_size=1)
        self.veggie_compub = rospy.Publisher('VeggieCommand_ros', PoseStamped, queue_size=1)
        self.compub = {'bottle_points':self.bottle_compub, 'meat_points':self.meat_compub, 'potato_points':self.potato_compub, 'tomato_points':self.tomato_compub, \
                        'tomato2_points':self.tomato2_compub, 'veggie_points':self.veggie_compub}
        self.pub_test_temp = rospy.Publisher('test_teapot', PointCloud2, queue_size=1)

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

    def pub_vel_command(self, dirt, rot_r):
        #first rotate to 0 
        euler = tf.transformations.euler_from_quaternion([rot_r[0], rot_r[1], rot_r[2], rot_r[3]])
        vel_msg = Twist()
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = euler[2]
        self.velocity_publisher.publish(vel_msg)

        #cmd
        if dirt == np.array([0, 1]):
            linear = [1]
            angular = [0]
        elif dirt == np.array([0, -1]):
            linear = [-1]
            angular = [0]
        elif dirt == np.array([-1, 0]):
            angular = [-90, 0]
            linear = [0, 1.0]
        elif dirt == np.array([0, -1]):
            angular = [90, 0]
            linear = [0, 1]
        elif dirt == np.array([-1, -1]):
            linear = [-1, 0, 1]
            angular = [0, -90, 0]
        elif dirt == np.array([-1, 1]):
            linear = [1, 0, 1]
            angular = [0, -90, 0]
        elif dirt == np.array([1, 1]):
            linear = [1, 0, 1]
            angular = [0, 90, 0]
        elif dirt == np.array([1, -1]):
            linear = [-1, 0, 1]
            angular = [0, 90, 0]
        for i, linear_ in enumerate(linear):
            vel_msg = Twist()
            vel_msg.linear.x = linear*self.grid_map.info.resolution
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0
            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = angular[i]
            self.velocity_publisher.publish(vel_msg)
            time.sleep(1/100.0)

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

    def callback(self):
        print('points recevied')
        height = self.grid_map.info.height
        width = self.grid_map.info.width
        data = -1*np.ones(width*height)

        level3_objects = []
        num_objs = np.unique(self.id_flags).shape[0]
        print(np.unique(self.id_flags))
        for i in range(1, num_objs):
            print(self.ar_flags.shape[0], self.id_flags.shape[0], self.map_points.shape[0])
            assert self.ar_flags.shape[0] == self.id_flags.shape[0] == self.map_points.shape[0]
            ar_flags = self.ar_flags[self.id_flags == i]
            if ar_flags[ar_flags == 3].shape[0] > ar_flags.shape[0]*0.5:
                level3_objects.append(i)
        print(level3_objects)
        if len(level3_objects) == 0:
            print('no object in level 3')

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

        level3_objects = level3_objects
        level3_flags = np.zeros(len(level3_objects))
        grid_map = self.grid_map.data
        grid_map = np.array(grid_map).reshape((height, width))
        print(grid_map.shape)
        indx_h = int((trans_h[0] - origin_x)/self.grid_map.info.resolution)
        indy_h = int((trans_h[1] - origin_y)/self.grid_map.info.resolution)
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                grid_map[indx_h + i, indy_h + j] = 1

        objs_path = dict()
        objs_path['res'] = self.grid_map.info.resolution
        objs_path['origin'] = self.grid_map.info.origin
        while(len(level3_objects) > 0):
            print(level3_objects)
            path = self.check_nearest_obj(level3_objects, (src_x, src_y), r, grid_map) #[obj_idx, path, dst, i]
            if path is None:
                print('cost is too high')
                continue
            if path[1] is None or len(path[1]) == 0:
                print('No path find', path[3])
                obj_idx = path[0]
                level3_flags[obj_idx] = 1
                del level3_objects[obj_idx]
                continue

            obj_idx = path[0]
            print('obj_id', path[3], obj_idx)
            level3_flags[obj_idx] = 1
            del level3_objects[obj_idx]
            #send vel cmd go to target

            data = -1*np.ones(width*height)
            objs_path[path[3]] = []
            objs_path[path[3]].append(path[1])
            for stop in path[1]:
                stop = np.array(stop)
                src = np.array([src_x, src_y])
                dirt = stop - src
                # self.pub_vel_command(dirt, rot_r)
                src = stop
                data[width*stop[1] + stop[0]] = 100

            # #find nearest point in level 1 - 2
            cost = 1000000
            min_path = None
            map_points = self.map_points[self.ar_flags == 1]
            msg = self.xyzrgb_array_to_pointcloud2(map_points, np.ones(map_points.shape), stamp=rospy.get_rostime(), frame_id='test_teapot', seq=None)
            self.pub_test_temp.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                'test_teapot',
                                'map')

            # moving_stops, idxs = self.find_neareat_point_inlevel1(path[1][-1], map_points, r, np.array([origin_x, origin_y]))
            # grid_map[grid_map == -1] = 0
            # grid_map[grid_map > 0] = 1
            # objs_path[path[3]].append(moving_stops)

            indx_h = int((trans_h[0] - origin_x)/self.grid_map.info.resolution)
            indy_h = int((trans_h[1] - origin_y)/self.grid_map.info.resolution)
            moving_stops = [(indx_h, indy_h)]
            for pid, moving_stop in enumerate(moving_stops):        
                # msg = self.grid_map
                # msg.data = tuple(data)
                # msg.info.width = self.grid_map.info.width
                # msg.info.height = self.grid_map.info.height
                # msg.info.origin = self.grid_map.info.origin
                # self.grid_pub.publish(msg)
                # self.br.sendTransform((0, 0, 0),
                #                     [0, 0, 0, 1],
                #                     rospy.Time.now(),
                #                     "ar_grid_map",
                #                     'map')      
                planner = AstarPlanner(grid=grid_map)
                planned_path, cost = planner.search(path[1][-1], (int(moving_stop[0]), int(moving_stop[1])))
                print(moving_stop, len(planned_path), cost)
                min_path = planned_path
            objs_path[path[3]].append(min_path)

            #loop send vel cmd and update ar location
            for stop in min_path:
                stop = np.array(stop)
                src = path[1][-1]
                dirt = stop - src
                # self.pub_vel_command(dirt, rot_r)
                src = stop
                data[width*stop[1] + stop[0]] = 100
                # self.pub_pose(stop, self.frame_ids[path[3]])
            
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

            self.begin_check = 0
            #update occupancy map
            # assert self.map_points.shape[0] == self.id_flags.shape[0]
            # points = self.map_points[self.id_flags == 2]
            # if self.frame_ids[obj_id[1]] == 'teapot_points':
            # msg = self.xyzrgb_array_to_pointcloud2(points, np.ones(points.shape), stamp=rospy.get_rostime(), frame_id='test_teapot', seq=None)
            # self.pub_test_temp.publish(msg)
            # self.br.sendTransform((0, 0, 0),
            #                     tf.transformations.quaternion_from_euler(0, 0, 0),
            #                     rospy.Time.now(),
            #                     'test_teapot',
            #                     'map')
            # print(grid_map.shape)
            src = min_path[0]
            dst = min_path[-1]
            src_x = dst[0]
            src_y = dst[1]
            diff = [dst[0] - src[0], dst[1] - src[1]]
            print(self.id_flags)
            print(self.id_flags[path[3]-1])
            for point in self.map_points[int(self.id_flags[path[3]-1]):int(self.id_flags[path[3]])]:
                indx = int((point[0] - origin_x)/self.grid_map.info.resolution)
                indy = int((point[1] - origin_y)/self.grid_map.info.resolution)
                grid_map[indx, indy] = 0
                grid_map[indx + diff[0], indy + diff[1]] = 1
            grid_map[grid_map > 0] = 100

            msg = self.grid_map
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'new_grid_map'
            msg.data = tuple(grid_map.reshape(-1))
            msg.info.width = self.grid_map.info.width
            msg.info.height = self.grid_map.info.height
            msg.info.origin = self.grid_map.info.origin
            self.new_grid_pub.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                [0, 0, 0, 1],
                                rospy.Time.now(),
                                "new_grid_map",
                                'map')
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
    

