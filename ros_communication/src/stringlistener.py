#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import pickle
import numpy as np

rospy.init_node('listener', anonymous=True)
file_name = rospy.get_param('~file_name')
data_ = []

def save_file():
    global data_
    data_ = np.array(data_)
    with open(file_name, 'wb') as fin:
        pickle.dump(data_, fin)

def callback(data):
#     global data_
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    data_.append(int(data.data.split(':')[1]))
    
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously

    rospy.Subscriber("SphereString", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    rospy.on_shutdown(save_file)

if __name__ == '__main__':
    listener()