#!/usr/bin/env python
import rospy
import message_filters
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CameraInfo

# def callback(data):
#     print(data) 
    
# def listener():

#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     rospy.init_node('image_listener', anonymous=True)

#     rospy.Subscriber("/kinect2/qhd/camera_info", CameraInfo, callback)

#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()
def world_to_camera(point)
    fy = h / ( 2 tan(43/2) )
    fx = w / 2 tan(57/2) 

def callback(image, wheel):
     bridge = CvBridge()
     rgb_frame = bridge.imgmsg_to_cv2(image, "bgr8") 
     print(rgb_frame.shape)
     for p in pc2.read_points(wheel, field_names = ("x", "y", "z","r","g","b"), skip_nans=True):
        print(p)


def listener():
    rospy.init_node('image_listener', anonymous=True)
    image_sub = message_filters.Subscriber('/kinect2/qhd/image_color',Image)
    wheel_points_sub = message_filters.Subscriber('wheel_world', PointCloud2)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, wheel_points_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    listener()