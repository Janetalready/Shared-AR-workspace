#!/usr/bin/env python
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point
#from pyquaternion import Quaternion
import numpy as np
from openpose_ros_msgs.msg import PersonDetection_3d
import tf
import Queue
from people_msgs.msg import Person, People


q = Queue.Queue()
pub = rospy.Publisher('cube_world', PointCloud2, queue_size=50)
pub_cube = rospy.Publisher('cube_center', PoseStamped, queue_size=50)
pub_person = rospy.Publisher('person_center', PoseStamped, queue_size=50)
pub_normal = rospy.Publisher('normal', PoseStamped, queue_size=50)
pub_people = rospy.Publisher('people', People, queue_size=10)
rospy.init_node('cube_world_pub', anonymous=True)
rate = rospy.Rate(10) # 10hz
tf_listener = tf.TransformListener()
br = tf.TransformBroadcaster()
FLOAT_EPS = np.finfo(np.float).eps
density = rospy.get_param('~density')
scale = rospy.get_param('~scale')
# print(rospy.get_param_names())
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

x_neck = 0 
x_nose = 0 
x_ankle = 0 
y_neck = 0 
y_nose = 0 
y_ankle = 0 
z_neck = 0
z_nose = 0 
z_ankle = 0
count = 1

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

def callback(cube, wheel, person):
    print('points received')
    global x_neck,x_nose,x_ankle,y_neck,y_nose,y_ankle,z_neck,z_nose,z_ankle,count
    if (((not np.isnan(person.left_knee.x)) and (not np.isnan(person.left_knee.y)) and (not np.isnan(person.left_knee.z)) and person.left_knee.confidence>0.5)  \
        and ((not np.isnan(person.right_knee.x)) and (not np.isnan(person.right_knee.y)) and (not np.isnan(person.right_knee.z)) and person.right_knee.confidence>0.5) and \
        ((not np.isnan(person.right_ankle.x)) and (not np.isnan(person.right_ankle.y)) and (not np.isnan(person.right_ankle.z)) and person.right_ankle.confidence>0.5) ):
            x_neck = person.left_knee.x;
            y_neck = person.left_knee.y;
            z_neck = person.left_knee.z;
            x_nose = person.right_knee.x;
            y_nose = person.right_knee.y;
            z_nose = person.right_knee.z;
            x_ankle = person.right_ankle.x;
            y_ankle = person.right_ankle.y;
            z_ankle = person.right_ankle.z;
    else:
        count += 1
        if count%10 == 0:
            x_neck = 0 
            x_nose = 0 
            x_ankle = 0 
            y_neck = 0 
            y_nose = 0 
            y_ankle = 0 
            z_neck = 0
            z_nose = 0 
            z_ankle = 0
            count = 1

  
  
    a = float(x_neck - x_nose);
    b = float(y_neck - y_nose);
    c = float(z_neck - z_nose);
    d = float(-(x_nose - x_ankle));
    e = float(-(y_nose - y_ankle));
    f = float(-(z_nose - z_ankle));
    normal = np.cross(np.array([a,b,c]),np.array([d,e,f]))
    length = np.linalg.norm(normal)
    x_center = (x_neck+x_nose)/2;
    y_center = (y_neck+y_nose)/2;
    z_center = (z_neck+z_nose)/2;
    print(length)
    print([x_center, y_center, z_center])
    print(normal)
    if not length==0:
        new_pose = [-cube.pose.position.y*normal[0]*5/length+x_center, -cube.pose.position.z*normal[1]*0.8/length+y_center, -cube.pose.position.x*normal[2]*0.8/length+z_center]
        
        points = []
        colors = []
        #quaternion = Quaternion(array=np.array([cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z, cube.pose.orientation.w]))
        obj_rotation = quat2mat([cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z, cube.pose.orientation.w])
        for idx, p in enumerate(pc2.read_points(wheel, field_names = ("x", "y", "z"), skip_nans=True)):
            if idx%density==0:
                point = np.array([p[0],p[1],p[2]])
                point = np.array(point)
                points.append(obj_rotation.dot(point.transpose()).transpose())
                colors.append([0.0,1.0,0.0])
        points = np.array(points)
        points[:,0] = (points[:,0]-points[:,0].mean())/scale + x_center + cube.pose.position.y*0.5#new_pose[0]
        points[:,1] = (points[:,1]-points[:,1].mean())/scale + y_center - cube.pose.position.z - 1#new_pose[1] - 0.5
        points[:,2] = (points[:,2]-points[:,2].mean())/scale + z_center - cube.pose.position.x*0.5#new_pose[2]     
        cube_x = (points[:,0].mean())/scale + x_center + cube.pose.position.y*0.5#new_pose[0]
        cube_y = (points[:,1].mean())/scale + y_center - cube.pose.position.z - 1#new_pose[1] - 0.5
        cube_z = (points[:,2].mean())/scale + z_center - cube.pose.position.x*0.5#new_pose[2]      
        colors = np.array(colors)
        msg = xyzrgb_array_to_pointcloud2(points, colors, stamp=rospy.get_rostime(), frame_id='cube_points', seq=None)
        pub.publish(msg)
        #q.put(msg)
        msg = PoseStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'cube_center'
        msg.pose.position.x = cube_x
        msg.pose.position.y = cube_y
        msg.pose.position.z = cube_z
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = 0
        msg.pose.orientation.z = 0
        msg.pose.orientation.w = 1
        pub_cube.publish(msg)
        br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "cube_center",
                            '/kinect2_ir_optical_frame')

        msg = PoseStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'person_center'
        msg.pose.position.x = x_center
        msg.pose.position.y = y_center
        msg.pose.position.z = z_center
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = 0
        msg.pose.orientation.z = 0
        msg.pose.orientation.w = 1
        pub_person.publish(msg)
        br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "person_center",
                            '/kinect2_ir_optical_frame')

        msg = PoseStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'normal'
        msg.pose.position.x = normal[0]
        msg.pose.position.y = normal[1]
        msg.pose.position.z = normal[2]
        msg.pose.orientation.x = 0
        msg.pose.orientation.y = 0
        msg.pose.orientation.z = 0
        msg.pose.orientation.w = 1
        pub_normal.publish(msg)
        br.sendTransform((0, 0, 0),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "normal",
                            '/kinect2_ir_optical_frame')

        br.sendTransform((0, 0, 0),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "cube_points",
                        '/kinect2_ir_optical_frame')
        
        msg_p = Person()
        msg_p.name = 'person'
        point = Point()
        point.x = x_center
        point.y = y_center
        point.z = z_center
        msg_p.position = point
        point.x = normal[0]/length*0.1
        point.y = normal[1]/length*0.1
        point.z = normal[2]/length*0.1
        msg_p.velocity = point

        msg_c = Person()
        msg_c.name = 'cube'
        point = Point()
        point.x = cube_x
        point.y = cube_y
        point.z = cube_z
        msg_c.position = point
        idx = np.argmax(normal)
        normal[idx] = -normal[idx]
        point.x = normal[0]/length*0.1
        point.y = normal[1]/length*0.1
        point.z = normal[2]/length*0.1
        msg_c.velocity = point

        msg = People()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'people'
        msg.people = [msg_p, msg_c]
        pub_people.publish(msg)
        br.sendTransform((0, 0, 0),
                                tf.transformations.quaternion_from_euler(0, 0, 0),
                                rospy.Time.now(),
                                "people",
                                '/kinect2_ir_optical_frame')
    else:
        new_pose = [-cube.pose.position.y*normal[0]*5/length+x_center, -cube.pose.position.z*normal[1]*0.8/length+y_center, -cube.pose.position.x*normal[2]*0.8/length+z_center]
        
        points = []
        colors = []
        #quaternion = Quaternion(array=np.array([cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z, cube.pose.orientation.w]))
        obj_rotation = quat2mat([cube.pose.orientation.x, cube.pose.orientation.y, cube.pose.orientation.z, cube.pose.orientation.w])
        for idx, p in enumerate(pc2.read_points(wheel, field_names = ("x", "y", "z"), skip_nans=True)):
            if idx%density==0:
                point = np.array([p[0],p[1],p[2]])
                point = np.array(point)
                points.append(obj_rotation.dot(point.transpose()).transpose())
                colors.append([0.0,1.0,0.0])
        points = np.array(points)
        points[:,0] = (points[:,0]-points[:,0].mean())/scale + x_center + 100#new_pose[0]
        points[:,1] = (points[:,1]-points[:,1].mean())/scale + y_center - 100#new_pose[1] - 0.5
        points[:,2] = (points[:,2]-points[:,2].mean())/scale + z_center - 100#new_pose[2]     
        colors = np.array(colors)
        msg = xyzrgb_array_to_pointcloud2(points, colors, stamp=rospy.get_rostime(), frame_id='cube_points', seq=None)
        pub.publish(msg)
        #q.put(msg)
        br.sendTransform((0, 0, 0),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        rospy.Time.now(),
                        "cube_points",
                        '/kinect2_ir_optical_frame')


def listener():
    cube_sub = message_filters.Subscriber('CubePose_ros',PoseStamped)
    wheel_points_sub = message_filters.Subscriber('cloud_pcd', PointCloud2)
    human_sub = message_filters.Subscriber('/openpose_ros/skeleton_3d/detected_poses_keypoints_3d', PersonDetection_3d)

    ts = message_filters.ApproximateTimeSynchronizer([cube_sub, wheel_points_sub, human_sub], 10, 1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
