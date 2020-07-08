#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np

pub = rospy.Publisher('cube_points', PointCloud2, queue_size=10)
rospy.init_node('cube_pub', anonymous=True)
rate = rospy.Rate(10) # 10hz

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

def callback(data):
    points = []
    colors = []
    for p in pc2.read_points(data, field_names = ("x", "y", "z"), skip_nans=True):
        points.append([p[0],p[1],p[2]])
        colors.append([1.0,0.0,0.0])
    points = np.array(points)
    points[:,0] = points[:,0]-points[:,0].mean()
    points[:,1] = points[:,1]-points[:,1].mean()
    points[:,2] = points[:,2]-points[:,2].mean()
    colors = np.array(colors)
    msg = xyzrgb_array_to_pointcloud2(points, colors, stamp=rospy.get_rostime(), frame_id='wheel', seq=None)
    pub.publish(msg)

    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def listener():

    rospy.Subscriber("cloud_pcd", PointCloud2, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()