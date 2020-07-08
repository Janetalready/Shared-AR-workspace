#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
import tf
import numpy as np
from scipy.spatial.transform import Rotation as R

class CommandPub:
    def __init__(self):
        rospy.init_node('command listener', anonymous=True)
        self.pub = rospy.Publisher('TeapotCommand', PoseStamped, queue_size=1)
        self.tflistener = tf.TransformListener()

    def callback(self, camera):
        print(camera.header.frame_id)
        try:
            (trans, rot) = self.tflistener.lookupTransform('ar_frame', 'map', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
        print('command received')

        pose = np.array([camera.pose.position.x, camera.pose.position.y, camera.pose.position.z])
        r = R.from_quat([rot[0], rot[1], rot[2], rot[3]])
        quat = tf.transformations.quaternion_from_euler(0, -90, 0)
        pose = r.apply(pose) + np.array(trans)
        print(pose)
        msg = PoseStamped()	
        header = Header()
        msg.header.stamp = rospy.get_rostime()
        timestamp = rospy.get_rostime()
        header.seq = 0
        header.stamp = timestamp
        msg.header = header
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[2]
        msg.pose.position.z = -pose[1]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        self.pub.publish(msg)

    def listener(self):
        rospy.Subscriber("TeapotCommand_ros", PoseStamped, self.callback)
        rospy.spin()

if __name__ == '__main__':
    commander = CommandPub()
    commander.listener()