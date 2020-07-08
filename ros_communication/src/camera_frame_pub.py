#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

pub = rospy.Publisher('CameraFrame_ros', Image, queue_size=10)
bridge = CvBridge()
def callback(camera):
    print('received image of type: "%s"' % camera.format)
    np_arr = np.fromstring(camera.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    cv2.imshow('cv_img', image_np)
    cv2.waitKey(20)

    msg = Image()	
    header = Header()
    timestamp = rospy.get_rostime()
    header.seq = 0
    header.stamp = timestamp
    header.frame_id = 'camera_frame'
    msg = bridge.cv2_to_imgmsg(image_np, encoding="rgb8")
    msg.header = header
    pub.publish(msg)

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("CameraFrame", CompressedImage, callback)

    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    listener()