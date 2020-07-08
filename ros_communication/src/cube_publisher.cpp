#include "ros/ros.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/transform_broadcaster.h"
#include "message_filters/subscriber.h"
#include <message_filters/synchronizer.h>
#include "message_filters/sync_policies/approximate_time.h"

using namespace message_filters;
ros::Publisher  cube_pub;
void chatterCallback(const geometry_msgs::PoseStampedConstPtr& cube)
{
  geometry_msgs::PoseStamped cube_msg;
  std_msgs::Header header;
	ros::Time timestamp = ros::Time::now();
	header.seq = 0;
	header.stamp = timestamp;
	cube_msg.header = header;
  cube_msg.pose.position.x = cube->pose.position.x;
  cube_msg.pose.position.y = cube->pose.position.y;
  cube_msg.pose.position.z = cube->pose.position.z;
  cube_msg.pose.orientation.x = cube->pose.orientation.x;
  cube_msg.pose.orientation.y = cube->pose.orientation.y;
  cube_msg.pose.orientation.z = cube->pose.orientation.z;
  cube_msg.pose.orientation.w = cube->pose.orientation.w;
  cube_pub.publish(cube_msg);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "sphere_listener");
  ros::NodeHandle n;
  cube_pub = n.advertise<geometry_msgs::PoseStamped>("SpherePose_ros", 10);
  ros::Subscriber sub = n.subscribe("SpherePose", 10, chatterCallback);
 
  ros::spin();

  return 0;
}

