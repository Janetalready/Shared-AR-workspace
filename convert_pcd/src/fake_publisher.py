#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
import tf

obj = "bunny"
pub = rospy.Publisher("BunnyCommand", PoseStamped, queue_size=10)
flag = 1
def callback(camera):
    key = raw_input('key input:')
    keys = key.split()
    msg = PoseStamped()	
    header = Header()
    msg.header.stamp = rospy.get_rostime()
    timestamp = rospy.get_rostime()
    header.seq = 0
    header.stamp = timestamp
    msg.header = header
    
    print(type(key[0]))
    msg.pose.position.x = float(keys[0])
    msg.pose.position.y = float(keys[1])
    msg.pose.position.z = float(keys[2])
    msg.pose.orientation.x = 0
    msg.pose.orientation.y = 0
    msg.pose.orientation.z = 0
    msg.pose.orientation.w = 1
    # if key == 'i':
    #     msg.pose.position.x = camera.pose.position.x + 1
    #     pub.publish(msg)
    # elif key == 'k':
    #     msg.pose.position.x = camera.pose.position.x - 1
    #     pub.publish(msg)
    # elif key == 'j':
    #     msg.pose.position.y = camera.pose.position.y + 1
    #     pub.publish(msg)
    # elif key == 'l':
    #     msg.pose.position.y = camera.pose.position.y - 1
    #     pub.publish(msg)
    # elif key == 'u':
    #     msg.pose.position.z = camera.pose.position.z + 1
    #     pub.publish(msg)
    # elif key == 'o':
    #     msg.pose.position.z = camera.pose.position.z + 1
    pub.publish(msg)
        

def listener():
    rospy.init_node('command_listener', anonymous=True)

    rospy.Subscriber("BunnyPose", PoseStamped, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()