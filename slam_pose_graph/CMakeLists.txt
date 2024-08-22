cmake_minimum_required(VERSION 3.0.2)
project(slam_pose_graph)

add_compile_options(-std=c++11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

find_package(catkin REQUIRED COMPONENTS roscpp
                                        rospy
                                        std_msgs)

catkin_python_setup()

# catkin_package(INCLUDE_DIRS     include
catkin_package(CATKIN_DEPENDS   roscpp
                                rospy
                                std_msgs)

install(DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        FILES_MATCHING PATTERN "*.launch"
        )

catkin_install_python(PROGRAMS
        src/GraphDrawer.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# include_directories(include ${catkin_INCLUDE_DIRS})
