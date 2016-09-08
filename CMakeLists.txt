cmake_minimum_required(VERSION 2.8.3)
project(HMPy)

find_package(catkin REQUIRED COMPONENTS message_generation
  rospy
  roscpp
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  roscpp

)

#catkin_python_setup()

