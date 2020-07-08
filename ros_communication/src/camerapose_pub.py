#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header

pub = rospy.Publisher('CameraPose_ros', PoseStamped, queue_size=1)
def callback(camera):
    msg = PoseStamped()	
    header = Header()
    msg.header.stamp = rospy.get_rostime()
    timestamp = rospy.get_rostime()
    header.seq = 0
    header.stamp = timestamp
    msg.header = header
    msg.pose = camera.pose
    pub.publish(msg)

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("CameraPose", PoseStamped, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()