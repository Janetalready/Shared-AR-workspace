#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped

def talker():
    pub = rospy.Publisher('SpherePose', PoseStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        cube_msg = PoseStamped()
  	header = Header()
  	timestamp = rospy.get_rostime()
  	header.seq = 0
  	header.stamp = timestamp
  	cube_msg.header = header
  	cube_msg.pose.position.x = 1.0
  	cube_msg.pose.position.y = 0.0
  	cube_msg.pose.position.z = -0.7
  	cube_msg.pose.orientation.x = 0
  	cube_msg.pose.orientation.y = 0
  	cube_msg.pose.orientation.z = 0
  	cube_msg.pose.orientation.w = 1
  	pub.publish(cube_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
