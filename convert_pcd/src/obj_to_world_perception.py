#!/usr/bin/env python
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point, PointStamped
#from pyquaternion import Quaternion
import numpy as np
# from openpose_ros_msgs.msg import PersonDetection_3d
import tf
import Queue
# from people_msgs.msg import Person, People
from scipy.spatial.transform import Rotation as R
import struct
import ctypes


q = Queue.Queue()
rospy.init_node('cube_world_pub', anonymous=True)
frame_id = rospy.get_param('~frame_id')
link_name = rospy.get_param('~link_name')
pub = rospy.Publisher('cube_world', PointCloud2, queue_size=1)
# pub_cube = rospy.Publisher('cube_center', PoseStamped, queue_size=50)
# pub_person = rospy.Publisher('person_center', PoseStamped, queue_size=50)
# pub_normal = rospy.Publisher('normal', PoseStamped, queue_size=50)
# pub_people = rospy.Publisher('people', People, queue_size=10)
rate = rospy.Rate(80) # 10hz
tf_listener = tf.TransformListener()
br = tf.TransformBroadcaster()
FLOAT_EPS = np.finfo(np.float).eps
density = rospy.get_param('~density')
scale = rospy.get_param('~scale')
# print(rospy.get_param_names())
rot = R.from_euler('yz', [-90, 180], degrees=True)
rot2 = R.from_euler('x', 90, degrees = True)
def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
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
        print(N)
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

def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < FLOAT_EPS:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def callback(cube, sphere_cloud):
    print(frame_id, 'points received')

    points = []
    colors = []
    if frame_id == 'teapot_points' or frame_id == 'table_points' or frame_id == 'meat_points' or frame_id == 'bottle_points' \
        or frame_id == 'potato_points' or frame_id == 'tomato_points' or frame_id == 'tomato2_points' or frame_id == 'veggie_points':
        print(frame_id, density)
        for idx, p in enumerate(pc2.read_points(sphere_cloud, field_names = ("x", "y", "z", "rgb"), skip_nans=True)):
            if idx%density==0:
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
    else:
        for idx, p in enumerate(pc2.read_points(sphere_cloud, field_names = ("x", "y", "z"), skip_nans=True)):
            if idx%density==0:
                point = np.array([p[0],p[1],p[2]])
                if frame_id == "cylinder_points":
                    point = r.apply(point)
                points.append(point)
                if frame_id == 'sphere1_points' or frame_id == 'sphere4_points' or frame_id == 'sphere5_points' or frame_id == 'sphere6_points':
                    colors.append([50.0/255,205.0/255,50.0/255])
                elif frame_id == 'sphere2_points' or frame_id == 'sphere7_points':
                    colors.append([102/255.0,221/255.0,229/255.0])
                elif frame_id == 'sphere3_points':
                    colors.append([218/255.0,165/255.0,32/255.0])
                elif frame_id == 'bunny_points':
                    colors.append([0, 172.0/255, 223.0/255])
                else:
                    colors.append([1,1,0])
    points = np.array(points)
    points[:,0] = points[:,0]-points[:,0].mean()
    points[:,1] = points[:,1]-points[:,1].mean()
    points[:,2] = points[:,2]-points[:,2].mean()
    points = points/scale
    if frame_id == 'teapot_points' or frame_id == 'table_points' or frame_id == 'meat_points' or frame_id == 'bottle_points' \
        or frame_id == 'potato_points' or frame_id == 'veggie_points':
        points = rot.apply(points)
    else:
        points = rot2.apply(points)
    cube_x = cube.pose.position.x
    cube_y = cube.pose.position.z
    cube_z = cube.pose.position.y
    msg = xyzrgb_array_to_pointcloud2(points, np.array(colors), stamp=rospy.get_rostime(), frame_id=frame_id, seq=None)
    pub.publish(msg)
    euler = tf.transformations.euler_from_quaternion([cube.pose.orientation.x, -cube.pose.orientation.z, cube.pose.orientation.y, cube.pose.orientation.w])
    br.sendTransform((cube_x, -cube_y, cube_z),
                       tf.transformations.quaternion_from_euler(0, euler[2], euler[1]),
                        rospy.Time.now(),
                        frame_id,
                        'ar_frame')

    br.sendTransform((cube_x, -cube_y, cube_z),
                        tf.transformations.quaternion_from_euler(euler[0], euler[2], euler[1]),
                        rospy.Time.now(),
                        link_name,
                        'ar_frame')
    rate.sleep()


def listener():
    sphere_sub = message_filters.Subscriber('SpherePose_ros',PoseStamped)
    sphere_points_sub = message_filters.Subscriber('cloud_pcd', PointCloud2)

    ts = message_filters.ApproximateTimeSynchronizer([sphere_sub, sphere_points_sub], 10, 1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
