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
import ctypes
import struct
import Queue
from threading import Thread, Lock
from nav_msgs.msg import OccupancyGrid

class MapUpdate:
    def __init__(self):
        rospy.init_node('map_listener', anonymous=True)
        self.pub = rospy.Publisher('map_update', PointCloud2, queue_size=5)
        self.grid_pub = rospy.Publisher('new_map', OccupancyGrid, queue_size=5)
        self.br = tf.TransformBroadcaster()
        self.q = Queue.Queue(maxsize=5)
        self.map_sub = rospy.Subscriber('map_cloud_pcd',PointCloud2, self.callback)
        self.grid_sub = rospy.Subscriber('map',OccupancyGrid, self.grid_callback)
        # data containers and its mutexes
        self.colors = None
        self.points = None
        self.map_mutex = Lock()
        self.grid_mutex = Lock()
        self.tflistener = tf.TransformListener()

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

    def grid_callback(self, grid):
        print('grid received')
        self.grid_msg = grid

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

    def callback(self, voxel):
        print('points recevied')
        rate = rospy.Rate(10)
        points = []
        colors = []
        self.flag = 1
        for pid, p in enumerate(pc2.read_points(voxel, field_names = ("x", "y", "z","rgb"), skip_nans=True)):
            # if pid%50 != 0:
            #     continue
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
        points = np.array(points)    
        colors = np.array(colors)
        self.points = points
        self.colors = colors
        msg = self.pointcloud2withrgb(points, colors, stamp=rospy.get_rostime(), frame_id='map_update', seq=None)
        self.pub.publish(msg)
        self.br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "map_update",
                            'map')
        self.flag = 0
        rate.sleep()

    def check_q(self):
        if self.flag == 0:
            msg = self.xyzrgb_array_to_pointcloud2(self.points, self.colors, stamp=rospy.get_rostime(), frame_id='map_update', seq=None)
            self.pub.publish(msg)
            self.br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                "map_update",
                                'map')

        

if __name__ == '__main__':
    data = MapUpdate()
    try:
        while not rospy.is_shutdown():
            data.map_mutex.acquire()
            if (data.points is not None):
                msg = data.pointcloud2withrgb(data.points, data.colors, stamp=rospy.get_rostime(), frame_id='map_update', seq=None)
                data.pub.publish(msg)
                data.br.sendTransform((0, 0, 0),
                                    tf.transformations.quaternion_from_euler(0, 0, 0),
                                    rospy.Time.now(),
                                    "map_update",
                                    'map')
                print('sent')
                data.map_mutex.release()
            else:
                data.map_mutex.release()

    except KeyboardInterrupt:
        print("Shutting down")
