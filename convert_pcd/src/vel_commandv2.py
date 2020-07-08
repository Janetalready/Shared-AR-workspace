#!/usr/bin/env python
# license removed for brevity
__author__ = 'fiorellasibona'
import rospy
import math

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler
import tf
import numpy as np
from std_msgs.msg import String
import pickle

class MoveBaseSeq():

    def __init__(self):

        rospy.init_node('move_base_sequence')
        self.br = tf.TransformBroadcaster()
        self.command_frame = {'bunny_points':'BunnyCommand_ros', 'table_points':'TableCommand_ros', \
        'teapot_points':'TeapotCommand_ros', 'sphere6_points':'Sphere6Command_ros', 'tomato_points':'TomatoCommand_ros', \
        'veggie_points':'VeggieCommand_ros'}
        self.bunny_compub = rospy.Publisher('BunnyCommand_ros', PoseStamped, queue_size=1)
        self.table_compub = rospy.Publisher('TableCommand_ros', PoseStamped, queue_size=1)
        self.teapot_compub = rospy.Publisher('TeapotCommand_ros', PoseStamped, queue_size=1)
        self.sphere6_compub = rospy.Publisher('Sphere6Command_ros', PoseStamped, queue_size=1)
        self.tomato_compub = rospy.Publisher('TomatoCommand_ros', PoseStamped, queue_size=1)
        self.veggie_compub = rospy.Publisher('VeggieCommand_ros', PoseStamped, queue_size=1)
        self.compub = {'bunny_points':self.bunny_compub, 'table_points':self.table_compub, 
        'teapot_points':self.teapot_compub, 'sphere6_points':self.sphere6_compub, 'tomato_points':self.tomato_compub, \
        'veggie_points':self.veggie_compub}
        self.frame_ids = {1:'veggie_points', 2: 'tomato_points', 7:'tomato_points', 3:'veggie_points'}
        self.veggie_hit = False
        self.tomato_hit = False
        # self.hit_ids = {1:self.veggie_hit, 2:self.tomato_hit}
        self.grid_sub = rospy.Subscriber('map', OccupancyGrid, self.grid_callback, queue_size=1)
        self.veggiehit_sub = rospy.Subscriber('VeggieHit', String, self.veggiehit_callback, queue_size=1)
        self.tomatohit_sub = rospy.Subscriber('TomatoHit', String, self.tomatohit_callback, queue_size=1)
        self.grid_pub = rospy.Publisher('navigation_map', OccupancyGrid, queue_size=1)
        self.grid_map = None
        while True:
            if self.grid_map is not None:
                print('map received')
                break
        # self.goal_pub = rospy.Publisher('quad_pos', PointStamped, queue_size=10)
        with open('/home/vcla/Workspace/turtle_ws/src/convert_pcd/src/pathv2.p', 'rb') as f:
            path_plan = pickle.load(f)
        self.width = 217 
        self.heigth = 202
        self.data = np.zeros(self.width*self.heigth)
        objects = [1, 2]
        origin_x = path_plan['origin'].position.x
        origin_y = path_plan['origin'].position.y
        self.orientation = path_plan['origin'].orientation
        res = path_plan['res']
        points_seq = []
        yaweulerangles_seq = []
        self.dirct = []
        self.object_ids = []
        self.stops = []
        self.pub_flags = []
        for idx in objects:
            self.goal_id = idx
            reach_plan = path_plan[idx][0]
            src_x = reach_plan[0][0]*res + origin_x
            src_y = reach_plan[0][1]*res + origin_y
            if idx == 1:
                stop_ids = [8, len(reach_plan)/3*2, len(reach_plan)/3*2 + 3,len(reach_plan)-1]
            else:
                stop_ids = [len(reach_plan)-1]
            for stop_id in stop_ids: #range(1, len(reach_plan), 5):
            # for stop_id, stop in enumerate(reach_plan[1:]):
                stop = reach_plan[stop_id]
                x = (stop[0] + 7)*res + origin_x
                y = (stop[1] + 5)*res + origin_y
                dirct = np.array([x, y]) - np.array([src_x, src_y])
                self.dirct.append(dirct)
                src_x, src_y =  x, y
                yaw = self.cal_yaw(dirct)
                points_seq += [x, y, 0]
                yaweulerangles_seq += [yaw]
                self.object_ids.append(idx)
                self.stops.append(reach_plan[stop_id: stop_id + 3])
                self.pub_flags.append(0)

            reach_plan = path_plan[idx][1]
            src_x = reach_plan[0][0]*res + origin_x
            src_y = reach_plan[0][1]*res + origin_y
           
            for stop_id in range(0, len(reach_plan), 10):
            # for stop_id, stop in enumerate(reach_plan[1:]):
                stop = reach_plan[stop_id]
                x = (stop[0] + 7)*res + origin_x
                y = (stop[1] + 5)*res + origin_y
                dirct = np.array([x, y]) - np.array([src_x, src_y])
                self.dirct.append(dirct)
                src_x, src_y =  x, y
                yaw = self.cal_yaw(dirct)
                points_seq += [x, y, 0]
                yaweulerangles_seq += [yaw]
                self.object_ids.append(idx)
                self.stops.append(reach_plan[stop_id: stop_id + 10])
                self.pub_flags.append(1)
            stop = reach_plan[-1]
            x = (stop[0])*res + origin_x
            y = (stop[1])*res + origin_y
            dirct = np.array([x, y]) - np.array([src_x, src_y])
            self.dirct.append([0, 0])
            src_x, src_y =  x, y
            yaw = self.cal_yaw(dirct)
            points_seq += [x, y, 0]
            yaweulerangles_seq += [yaw]
            self.object_ids.append(idx)
            self.stops.append([stop])
            self.pub_flags.append(1)

        # points_seq = [-2.54, 2.95, 0]
        # yaweulerangles_seq = [0]
        quat_seq = list()
        self.pose_seq = list()
        self.goal_cnt = 0
        for yawangle in yaweulerangles_seq:
            quat_seq.append(Quaternion(*(quaternion_from_euler(0, 0, yawangle*math.pi/180, axes='sxyz'))))
        n = 3
        points = [points_seq[i:i+n] for i in range(0, len(points_seq), n)]
        for point in points:
            self.pose_seq.append(Pose(Point(*point),quat_seq[n-3]))
            n += 1
        self.client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        wait = self.client.wait_for_server(rospy.Duration(5.0))
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
            return
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting goals achievements ...")
        self.movebase_client()
        
        # while True:
        #     self.pose_pub()

    def grid_callback(self, grid):
        self.grid_map = grid

    def veggiehit_callback(self, data):
        self.veggie_hit = True

    def tomatohit_callback(self, data):
        self.tomato_hit = True

    def cal_yaw(self, dirct):
        angle = np.arctan2(dirct[1], dirct[0])
        return angle*180/np.pi

    def active_cb(self):
        rospy.loginfo("Goal pose "+str(self.goal_cnt+1)+" is now being processed by the Action Server...")

    def feedback_cb(self, feedback):
        rospy.loginfo("Feedback for goal pose "+str(self.goal_cnt+1)+" received")

    def pose_pub(self, dirct, object_id):
        frame_id = self.frame_ids[object_id]
        self.br.sendTransform((self.pose_seq[self.goal_cnt].position.x, self.pose_seq[self.goal_cnt].position.y, -1),
                        [0, 0, 0, 1],
                        rospy.Time.now(),
                        self.command_frame[frame_id],
                        'map')
        # self.br.sendTransform((0, 0, 0),
        #                 [0, 0, 0, 1],
        #                 rospy.Time.now(),
        #                 self.command_frame[frame_id],
        #                 'map')
        p = PoseStamped()
        p.header.frame_id = self.command_frame[frame_id]
        p.header.stamp = rospy.Time.now()

        dirct  = dirct/np.linalg.norm(dirct)
        p.pose.position.x = self.pose_seq[self.goal_cnt].position.x + dirct[0]*0.001
        p.pose.position.y = self.pose_seq[self.goal_cnt].position.y + dirct[1]*0.001
        p.pose.position.z = 0.5
        p.pose.orientation.x = 0
        p.pose.orientation.y = 0
        p.pose.orientation.z = 0
        p.pose.orientation.w = 1
        self.compub[frame_id].publish(p)

    def pub_grid_map(self, goal_id):
        for stop in self.stops[goal_id]:
            self.data[(stop[1] + 5)*self.width + stop[0] + 7] = 100
        grid = self.grid_map
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = 'navigation_map'
        grid.data = tuple(self.data.reshape(-1))
        self.grid_pub.publish(grid)
        self.br.sendTransform((0, 0, 0), [0, 0, 0, 1], rospy.Time.now(), 'navigation_map', 'map')
        
        

    def done_cb(self, status, result):
        self.goal_cnt += 1
        if status == 2:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" received a cancel request after it started executing, completed execution!")

        if status == 3:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" reached") 
            print('tomato', self.tomato_hit)
            print('veggie', self.veggie_hit)
            if self.goal_cnt< len(self.pose_seq):
                idx = self.object_ids[self.goal_cnt - 1]
                skip_goal = False
                if idx == 1:
                    flag = self.veggie_hit
                elif idx == 2:
                    flag = self.tomato_hit
                else:
                    flag = False
                while(flag):
                    self.goal_cnt += 1
                    print(self.goal_cnt - 1, len(self.object_ids))
                    if self.goal_cnt - 1 > len(self.object_ids) - 1:
                        self.goal_cnt = len(self.pose_seq) - 1
                        break
                    print(idx, flag)
                    idx = self.object_ids[self.goal_cnt - 1]
                    if idx == 1:
                        flag = self.veggie_hit
                    elif idx == 2:
                        flag = self.tomato_hit
                    else:
                        flag = False
                    skip_goal = True
                if skip_goal:
                    while(self.goal_cnt - 1 < len(self.pub_flags) and self.pub_flags[self.goal_cnt - 1] == 0):
                        self.goal_cnt += 1
                next_goal = MoveBaseGoal()
                next_goal.target_pose.header.frame_id = "map"
                next_goal.target_pose.header.stamp = rospy.Time.now()
                next_goal.target_pose.pose = self.pose_seq[self.goal_cnt]
                rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
                rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
                self.client.send_goal(next_goal, self.done_cb, self.active_cb, self.feedback_cb)
                if self.pub_flags[self.goal_cnt - 1]:
                    count = 0
                    while(count < 10):     
                        self.pose_pub(self.dirct[self.goal_cnt - 1], self.object_ids[self.goal_cnt - 1])
                        self.pub_grid_map(self.goal_cnt-1)
                        count += 1
                # if self.object_ids[self.goal_cnt - 1] == 3:
                # raw_input('Enter:')
                
            else:
                rospy.loginfo("Final goal pose reached!")
                rospy.signal_shutdown("Final goal pose reached!")
                return

        if status == 4:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" was aborted by the Action Server")
            rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" aborted, shutting down!")
            return

        if status == 5:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" has been rejected by the Action Server")
            rospy.signal_shutdown("Goal pose "+str(self.goal_cnt)+" rejected, shutting down!")
            return

        if status == 8:
            rospy.loginfo("Goal pose "+str(self.goal_cnt)+" received a cancel request before it started executing, successfully cancelled!")

    def movebase_client(self):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now() 
        goal.target_pose.pose = self.pose_seq[self.goal_cnt]
        rospy.loginfo("Sending goal pose "+str(self.goal_cnt+1)+" to Action Server")
        rospy.loginfo(str(self.pose_seq[self.goal_cnt]))
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
        rospy.spin()

if __name__ == '__main__':
    try:
        MoveBaseSeq()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation finished.")