#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header

pub = rospy.Publisher('CameraForward_ros', PoseStamped, queue_size=10)
def callback(camera):
    msg = PoseStamped()	
    header = Header()
    timestamp = rospy.get_rostime()
    header.seq = 0
    header.stamp = timestamp
    msg.header = header
    msg.pose = camera.pose
    pub.publish(msg)

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("CameraForward", PoseStamped, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()